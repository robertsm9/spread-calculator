import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO


# =========================================================
# CONFIGURATION
# =========================================================

TICKER_SYMBOL = "BE"


# =========================================================
# FUNCTIONS
# =========================================================

def get_mid_price(row):
    bid = row["bid"]
    ask = row["ask"]
    last = row["lastPrice"]

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)

    if pd.notna(last):
        return round(last, 2)

    return 0.0


def get_option_row(calls_df, target_strike):
    matching_rows = calls_df[
        np.isclose(
            calls_df["strike"].astype(float),
            float(target_strike),
            atol=0.001
        )
    ]

    if matching_rows.empty:
        return None

    return matching_rows.iloc[0]


@st.cache_data(ttl=1800)
def fetch_current_price(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="5d")

    if history.empty:
        raise ValueError(
            f"Yahoo Finance did not return a current price for "
            f"{ticker_symbol}."
        )

    return round(float(history["Close"].dropna().iloc[-1]), 2)


@st.cache_data(ttl=1800)
def fetch_spread_term_structure(ticker_symbol, long_strike, short_strike):
    ticker = yf.Ticker(ticker_symbol)
    available_expirations = list(ticker.options)

    if not available_expirations:
        raise ValueError(
            f"No option expirations were found for {ticker_symbol}."
        )

    today = datetime.now().date()
    rows = []

    for expiration in available_expirations:
        try:
            expiry_date = datetime.strptime(
                expiration, "%Y-%m-%d"
            ).date()

            dte = (expiry_date - today).days

            if dte < 0:
                continue

            chain = ticker.option_chain(expiration)
            calls_df = chain.calls.copy()
            calls_df["mid"] = calls_df.apply(get_mid_price, axis=1)

            long_row = get_option_row(calls_df, long_strike)
            short_row = get_option_row(calls_df, short_strike)

            if long_row is None or short_row is None:
                continue

            long_premium = float(long_row["mid"])
            short_premium = float(short_row["mid"])

            if long_premium <= 0 or short_premium < 0:
                continue

            net_debit = long_premium - short_premium

            if net_debit <= 0:
                continue

            short_iv = float(short_row["impliedVolatility"]) * 100

            rows.append({
                "Expiration": expiration,
                "DTE": dte,
                "Call Bought Premium": round(long_premium, 2),
                "Call Sold Premium": round(short_premium, 2),
                "Implied Vol (Hi)": round(short_iv, 1),
                "Call Spread Cost": round(net_debit, 2)
            })

        except Exception:
            continue

    if not rows:
        raise ValueError(
            f"No usable {long_strike:.0f}/{short_strike:.0f} spread "
            f"quotes found for any expiration."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("DTE").reset_index(drop=True)

    return df


def build_excel_style_table(
    term_df,
    current_price,
    long_strike,
    short_strike,
    covered_call_strike
):
    width = short_strike - long_strike

    dtes = term_df["DTE"].tolist()
    ivs = term_df["Implied Vol (Hi)"].tolist()
    call_sold_premiums = term_df["Call Sold Premium"].tolist()
    spread_costs = term_df["Call Spread Cost"].tolist()

    marginal_ivs = []
    num_spreads_list = []
    total_profits = []
    combined_values = []
    returns = []
    returns_per_dte = []
    returns_per_marginal_dte = []

    prev_dte = 0
    prev_combined_value = current_price

    for i in range(len(term_df)):
        dte = dtes[i]
        iv = ivs[i]
        proceeds = call_sold_premiums[i]
        cost = spread_costs[i]

        if i == 0:
            marginal_iv = iv
        else:
            prior_iv = ivs[i - 1]
            prior_dte = dtes[i - 1]
            time_gap = dte - prior_dte

            if time_gap > 0:
                forward_var = (
                    (iv ** 2) * dte - (prior_iv ** 2) * prior_dte
                ) / time_gap
                marginal_iv = np.sqrt(max(forward_var, 0))
            else:
                marginal_iv = iv

        marginal_ivs.append(round(marginal_iv, 1))

        num_spreads = proceeds / cost if cost > 0 else 0
        num_spreads_list.append(round(num_spreads, 1))

        total_profit = num_spreads * width
        total_profits.append(round(total_profit, 1))

        combined_value = total_profit + covered_call_strike
        combined_values.append(round(combined_value, 1))

        ret = (combined_value - current_price) / current_price
        returns.append(round(ret * 100, 1))

        ret_per_dte = (ret / dte) if dte > 0 else 0
        returns_per_dte.append(round(ret_per_dte * 100, 1))

        marginal_dte = dte - prev_dte
        marginal_ret = (
            (combined_value - prev_combined_value) / current_price
        )
        ret_per_marginal_dte = (
            (marginal_ret / marginal_dte) if marginal_dte > 0 else 0
        )
        returns_per_marginal_dte.append(round(ret_per_marginal_dte * 100, 1))

        prev_dte = dte
        prev_combined_value = combined_value

    excel_rows = {
        f"Call bought: {long_strike:.0f}": term_df["Call Bought Premium"].tolist(),
        f"Call sold: {short_strike:.0f}": term_df["Call Sold Premium"].tolist(),
        f"Implied Volatility (Hi): {short_strike:.0f}": ivs,
        "Marginal IV": marginal_ivs,
        "DTE": dtes,
        "Call sold": call_sold_premiums,
        "Call spread cost": spread_costs,
        "Call spreads": num_spreads_list,
        "Profit/spread": [width] * len(term_df),
        "Total profit": total_profits,
        "Underlying share": [covered_call_strike] * len(term_df),
        "Combined value": combined_values,
        "Return": [f"{v}%" for v in returns],
        "Return/DTE": [f"{v}%" for v in returns_per_dte],
        "Return/Marginal DTE": [f"{v}%" for v in returns_per_marginal_dte]
    }

    display_df = pd.DataFrame(
        excel_rows,
        index=term_df["Expiration"].tolist()
    ).T

    return display_df


def convert_to_excel_bytes(
    display_df,
    current_price,
    ticker_symbol,
    long_strike,
    short_strike,
    covered_call_strike
):
    output = BytesIO()

    metadata_df = pd.DataFrame({
        "Field": [
            "Ticker",
            "Current Share Price",
            "Long Strike",
            "Short Strike",
            "Covered Call Strike",
            "Generated On"
        ],
        "Value": [
            ticker_symbol,
            current_price,
            long_strike,
            short_strike,
            covered_call_strike,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        display_df.to_excel(writer, sheet_name="Term Structure")
        metadata_df.to_excel(writer, sheet_name="Info", index=False)

    return output.getvalue()


# =========================================================
# DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Bloom Energy Spread Calculator",
    layout="wide"
)

st.title("Bloom Energy Spread Calculator")

ticker_symbol = TICKER_SYMBOL

try:
    live_price = fetch_current_price(ticker_symbol)

    st.metric("Live Price (Yahoo)", f"${live_price:.2f}")

    current_share_price = st.number_input(
        "Current Share Price ($)",
        min_value=0.01,
        value=float(live_price),
        step=1.0
    )

    covered_call_strike = st.number_input(
        "Call Sold (Covered Call Strike)",
        min_value=1.0,
        value=270.0,
        step=5.0
    )

    short_strike = st.number_input(
        "Hi (Call Sold Strike)",
        min_value=1.0,
        value=270.0,
        step=5.0
    )

    long_strike = st.number_input(
        "Lo (Call Bought Strike)",
        min_value=1.0,
        value=260.0,
        step=5.0
    )

    if short_strike <= long_strike:
        st.error("Hi strike must be greater than Lo strike.")
        st.stop()

    all_available_expirations = fetch_spread_term_structure(
        ticker_symbol=ticker_symbol,
        long_strike=long_strike,
        short_strike=short_strike
    )

    expiration_options = [
        f"{row['Expiration']} ({row['DTE']} DTE)"
        for _, row in all_available_expirations.iterrows()
    ]

    default_selection = (
        expiration_options[:8]
        if len(expiration_options) >= 8
        else expiration_options
    )

    selected_expiration_labels = st.multiselect(
        "Select Expirations to Display",
        options=expiration_options,
        default=default_selection
    )

    selected_dtes = [
        int(label.split("(")[1].replace(" DTE)", ""))
        for label in selected_expiration_labels
    ]

    term_df = all_available_expirations[
        all_available_expirations["DTE"].isin(selected_dtes)
    ].reset_index(drop=True)

    if term_df.empty:
        st.warning(
            "No expirations selected. Choose at least one from the "
            "list above."
        )
        st.stop()

    display_df = build_excel_style_table(
        term_df=term_df,
        current_price=current_share_price,
        long_strike=long_strike,
        short_strike=short_strike,
        covered_call_strike=covered_call_strike
    )

    st.markdown("---")
    st.markdown("### Full Term Structure")

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.markdown("---")

    excel_bytes = convert_to_excel_bytes(
        display_df=display_df,
        current_price=current_share_price,
        ticker_symbol=ticker_symbol,
        long_strike=long_strike,
        short_strike=short_strike,
        covered_call_strike=covered_call_strike
    )

    st.download_button(
        label="Generate & Download Excel Sheet",
        data=excel_bytes,
        file_name=f"BE_call_spread_{long_strike:.0f}_{short_strike:.0f}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"Error: {e}")
