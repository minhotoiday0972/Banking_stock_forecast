# src/data/data_collector.py
import vnstock
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Optional, List, Tuple

from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.database import get_database

logger = get_logger("data_collector")


class DataCollector:
    """Centralized data collection from vnstock"""

    def __init__(self):
        self.config = get_config()
        self.db = get_database()
        self.vnstock_obj = vnstock.Vnstock(source="TCBS")

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def fetch_ohlcv_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from vnstock"""
        try:
            stock = self.vnstock_obj.stock(symbol=ticker, source="TCBS")
            df = stock.quote.history(start=start_date, end=end_date, interval="1D")

            if df is None or df.empty:
                raise ValueError(f"No OHLCV data for {ticker}")

            # Standardize columns
            df["time"] = pd.to_datetime(df["time"])
            df = df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )

            # Data validation
            df = df[(df["Close"] > 0) & (df["Volume"] >= 0)]
            df["Ticker"] = ticker
            df = df[["time", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

            # Check for invalid data
            if df[["Open", "High", "Low", "Close"]].le(0).any().any():
                raise ValueError(f"Invalid OHLCV data for {ticker}")

            if df["Volume"].lt(0).any():
                raise ValueError(f"Invalid volume data for {ticker}")

            # Check data completeness
            expected_dates = pd.date_range(start=start_date, end=end_date, freq="B")
            if len(df) < len(expected_dates) * 0.8:
                logger.warning(f"Missing data for {ticker} (possible IPO or delisting)")

            logger.info(f"Fetched OHLCV data for {ticker}: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Could not fetch OHLCV for {ticker}: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def fetch_fundamental_data(
        self, ticker: str, period: str = "quarter"
    ) -> Optional[pd.DataFrame]:
        """Fetch fundamental data from vnstock with banking-specific features"""
        try:
            logger.info(f"Fetching fundamental data for {ticker}...")

            # Get raw data from vnstock
            ratios = self.vnstock_obj.stock(symbol=ticker, source="TCBS").finance.ratio(
                period=period, lang="vi"
            )

            if ratios is None or ratios.empty:
                raise ValueError(f"No fundamental data for {ticker}")

            logger.info(f"Raw data: {len(ratios)} rows, {len(ratios.columns)} columns")

            # Define ALL desired columns (general + banking specific)
            # Updated to include ALL available columns from vnstock
            desired_columns = [
                "quarter",
                "year",
                "roe",
                "roa",
                "price_to_earning",
                "price_to_book",
                "earning_per_share",
                "book_value_per_share",
                "interest_margin",
                "non_interest_on_toi",
                "bad_debt_percentage",
                "provision_on_bad_debt",
                "cost_of_financing",
                "equity_on_total_asset",
                "equity_on_loan",
                "cost_to_income",
                "loan_on_earn_asset",
                "loan_on_asset",
                "loan_on_deposit",
                "deposit_on_earn_asset",
                "bad_debt_on_asset",
                "credit_growth",
                "pre_provision_on_toi",
                "post_tax_on_toi",
                "equity_on_liability",
                "eps_change",
                "asset_on_equity",
                "liquidity_on_liability",
                "payable_on_equity",
                "cancel_debt",
                "book_value_per_share_change",
            ]

            # Check which columns are available
            available_columns = [
                col for col in desired_columns if col in ratios.columns
            ]
            logger.info(
                f"Available columns: {len(available_columns)}/{len(desired_columns)}"
            )

            if not available_columns:
                raise ValueError(f"No matching fundamental columns for {ticker}")

            # Select only available columns
            ratios_subset = ratios[available_columns].copy()

            # Apply column mapping
            column_mapping = {
                "roe": "ROE (%)",
                "roa": "ROA (%)",
                "price_to_earning": "P/E",
                "price_to_book": "P/B",
                "earning_per_share": "EPS (VND)",
                "book_value_per_share": "BVPS (VND)",
                # Banking specific mappings
                "interest_margin": "NIM (%)",
                "non_interest_on_toi": "Non_Interest_Income (%)",
                "bad_debt_percentage": "NPL (%)",
                "provision_on_bad_debt": "Provision_Coverage (%)",
                "cost_of_financing": "Cost_of_Funds (%)",
                "equity_on_total_asset": "Equity_Ratio (%)",
                "equity_on_loan": "Equity_on_Loan (%)",
                "cost_to_income": "CIR (%)",
                "loan_on_earn_asset": "Loan_on_Earning_Asset (%)",
                "loan_on_asset": "Loan_to_Asset (%)",
                "loan_on_deposit": "Loan_to_Deposit (%)",
                "deposit_on_earn_asset": "Deposit_Ratio (%)",
                "bad_debt_on_asset": "NPL_to_Asset (%)",
                "credit_growth": "Credit_Growth (%)",
                "pre_provision_on_toi": "Pre_Provision_ROA (%)",
                "post_tax_on_toi": "Post_Tax_ROA (%)",
            }

            # Rename columns
            ratios_mapped = ratios_subset.rename(columns=column_mapping)

            # Create time column
            ratios_mapped["time"] = pd.to_datetime(
                ratios_mapped["year"].astype(str)
                + "-"
                + ((ratios_mapped["quarter"] * 3 - 2).astype(str))
                + "-01"
            )
            ratios_mapped["Ticker"] = ticker

            logger.info(
                f"Processed fundamental data for {ticker}: {len(ratios_mapped)} rows, {len(ratios_mapped.columns)} columns"
            )
            return ratios_mapped

        except Exception as e:
            logger.error(f"Could not fetch fundamental data for {ticker}: {str(e)}")
            return None

    def fetch_vnindex_data(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch VN-Index data from CSV file"""
        try:
            vnindex_file = os.path.join(self.config.raw_dir, "vnindex_data.csv")

            if not os.path.exists(vnindex_file):
                logger.error(
                    "VN-Index CSV file not found. Please download from CafeF/HOSE."
                )
                return None

            df = pd.read_csv(vnindex_file)
            df["time"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"close": "VNINDEX"})
            df = df[(df["time"] >= start_date) & (df["time"] <= end_date)]
            df = df[["time", "VNINDEX"]]

            if df.empty:
                raise ValueError("VN-Index data is empty after filtering")

            logger.info(f"Loaded VN-Index data: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Could not fetch VN-Index data: {str(e)}")
            return None

    def collect_all_data(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """Collect all data for given tickers"""
        if tickers is None:
            tickers = self.config.tickers
        if start_date is None:
            start_date = self.config.get("data.start_date")
        if end_date is None:
            # Always use yesterday to ensure data availability
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        logger.info(f"Starting data collection for {len(tickers)} tickers")
        logger.info(f"Date range: {start_date} to {end_date}")

        # Clear existing tables
        self.db.clear_ticker_tables(tickers)

        # Collect VN-Index data
        vnindex_data = self.fetch_vnindex_data(start_date, end_date)
        if vnindex_data is not None:
            # Save to CSV
            vnindex_csv = os.path.join(self.config.raw_dir, "VNINDEX.csv")
            os.makedirs(self.config.raw_dir, exist_ok=True)
            vnindex_data.to_csv(vnindex_csv, index=False)

            # Save to database
            self.db.save_dataframe(vnindex_data, "VNINDEX")

        available_tickers = []
        failed_tickers = []

        # Process tickers in batches to avoid rate limits
        batch_size = self.config.get("data.batch_size", 3)
        delay_between_tickers = self.config.get("data.delay_between_tickers", 5)
        delay_between_batches = self.config.get("data.delay_between_batches", 10)

        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i : i + batch_size]
            logger.info(f"Processing batch: {batch_tickers}")

            for j, ticker in enumerate(batch_tickers):
                success = self._process_single_ticker(ticker, start_date, end_date)
                if success:
                    available_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)

                # Delay between tickers (except last ticker in batch)
                if j < len(batch_tickers) - 1:
                    logger.info(
                        f"⏳ Waiting {delay_between_tickers}s before next ticker..."
                    )
                    time.sleep(delay_between_tickers)

            # Delay between batches (except last batch)
            if i + batch_size < len(tickers):
                logger.info(f"⏳ Waiting {delay_between_batches}s before next batch...")
                time.sleep(delay_between_batches)

        # Create combined OHLCV data
        if available_tickers:
            self._create_combined_ohlcv(available_tickers)

        logger.info(
            f"Data collection completed. Available: {len(available_tickers)}, Failed: {len(failed_tickers)}"
        )
        return available_tickers, failed_tickers

    def _process_single_ticker(
        self, ticker: str, start_date: str, end_date: str
    ) -> bool:
        """Process single ticker data"""
        try:
            # Fetch OHLCV data
            ohlcv_data = self.fetch_ohlcv_data(ticker, start_date, end_date)
            if ohlcv_data is None:
                return False

            # Save OHLCV to CSV and database
            ohlcv_csv = os.path.join(self.config.raw_dir, f"{ticker}_ohlcv.csv")
            ohlcv_data.to_csv(ohlcv_csv, index=False)

            if not self.db.save_dataframe(ohlcv_data, f"{ticker}_OHLCV"):
                return False

            # Wait before fundamental request to avoid rate limit
            delay_between_requests = self.config.get("data.delay_between_requests", 2)
            logger.info(
                f"⏳ Waiting {delay_between_requests}s before fundamental request..."
            )
            time.sleep(delay_between_requests)

            # Fetch fundamental data
            fundamental_data = self.fetch_fundamental_data(ticker)
            if fundamental_data is None:
                logger.warning(f"No fundamental data for {ticker}, skipping")
                return False

            # Save fundamental to CSV and database
            fundamental_csv = os.path.join(
                self.config.raw_dir, f"{ticker}_fundamental.csv"
            )
            fundamental_data.to_csv(fundamental_csv, index=False)

            if not self.db.save_dataframe(fundamental_data, f"{ticker}_Fundamental"):
                return False

            logger.info(f"Successfully processed {ticker}")
            return True

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return False

    def _create_combined_ohlcv(self, available_tickers: List[str]):
        """Create combined OHLCV data for market analysis"""
        try:
            all_ohlcv = []

            for ticker in available_tickers:
                df = self.db.load_dataframe(f"{ticker}_OHLCV")
                if df is not None:
                    all_ohlcv.append(df)

            if all_ohlcv:
                combined_df = pd.concat(all_ohlcv, ignore_index=True)

                # Save to CSV
                combined_csv = os.path.join(self.config.raw_dir, "all_ohlcv.csv")
                combined_df.to_csv(combined_csv, index=False)

                # Save to database
                self.db.save_dataframe(combined_df, "All_OHLCV")

                logger.info(f"Created combined OHLCV data: {len(combined_df)} rows")

        except Exception as e:
            logger.error(f"Failed to create combined OHLCV data: {e}")


def main():
    """Main function for data collection"""
    collector = DataCollector()
    available, failed = collector.collect_all_data()

    print(f"\n=== Data Collection Summary ===")
    print(f"Available tickers: {available}")
    print(f"Failed tickers: {failed}")
    print(
        f"Success rate: {len(available)}/{len(available) + len(failed)} ({len(available)/(len(available) + len(failed))*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
