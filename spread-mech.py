import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
from io import BytesIO


# =========================================================
# CONFIGURATION
# =========================================================

TICKER_SYMBOL = "BE"
RISK_FREE_RATE = 0.03456  # matches the 3.456 shown in his sheet, as a decimal


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


def calculate_call_delta(spot, strike, dte, iv_percent, risk_free_rate):
    """
    Standard Black-Scholes call delta = N(d1).
    """

    if dte <= 0 or iv_percent <= 0 or spot <= 0 or strike <= 0:
        return 0.0

    iv = iv_percent / 100
    t = dte / 365

    d1 = (
        np.log(spot / strike)
        + (risk_free_rate + 0.5 * iv ** 2) * t
    ) / (iv * np.sqrt(t))

    delta = norm.cdf(d1)

    return round(float(delta), 5)


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
def fetch_spread_term_structure(ticker_symbol, long_strike, short_strike, covered_call_strike):
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
            covered_call_row = get_option_row(calls_df, covered_call_strike)

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
            long_iv = float(long_row["impliedVolatility"]) * 100

            if covered_call_row is not None:
                covered_call_iv = float(covered_call_row["impliedVolatility"]) * 100
            else:
                covered_call_iv = short_iv

            rows.append({
                "Expiration": expiration,
                "DTE": dte,
                "Call Bought Premium": round(long_premium, 2),
                "Call Sold Premium": round(short_premium, 2),
                "Implied Vol (Hi)": round(short_iv, 1),
                "Implied Vol (Lo)": round(long_iv, 1),
                "Implied Vol (Covered Call)": round(covered_call_iv, 1),
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
    ivs_lo = term_df["Implied Vol (Lo)"].tolist()
    ivs_cc = term_df["Implied Vol (Covered Call)"].tolist()
    call_sold_premiums = term_df["Call Sold Premium"].tolist()
    spread_costs = term_df["Call Spread Cost"].tolist()

    marginal_ivs = []
    num_spreads_list = []
    total_profits = []
    combined_values = []
    returns = []
    returns_per_dte = []
    returns_per_marginal_dte = []

    hi_deltas = []
    lo_deltas = []
    covered_call_deltas = []
    spread_delta_per_unit = []
    long_call_spread_total_delta = []
    covered_call_delta_contributions = []
    total_position_deltas = []

    prev_dte = None
    prev_return = None

    for i in range(len(term_df)):
        dte = dtes[i]
        iv_hi = ivs[i]
        iv_lo = ivs_lo[i]
        iv_cc = ivs_cc[i]
        proceeds = call_sold_premiums[i]
        cost = spread_costs[i]

        if i == 0:
            marginal_iv = iv_hi
        else:
            prior_iv = ivs[i - 1]
            prior_dte = dtes[i - 1]
            time_gap = dte - prior_dte

            if time_gap > 0:
                marginal_iv = (
                    (iv_hi * dte) - (prior_iv * prior_dte)
                ) / time_gap
            else:
                marginal_iv = iv_hi

        marginal_ivs.append(round(marginal_iv, 1))

        num_spreads = proceeds / cost if cost > 0 else 0
        num_spreads_list.append(round(num_spreads, 1))

        total_profit = num_spreads * width
        total_profits.append(round(total_profit, 1))

        combined_value = total_profit + covered_call_strike
        combined_values.append(round(combined_value, 1))

        ret = total_profit / current_price
        returns.append(round(ret * 100, 1))

        ret_per_dte = (ret / dte) if dte > 0 else 0
        returns_per_dte.append(round(ret_per_dte * 100, 1))

        if i == 0:
            ret_per_marginal_dte = ret_per_dte
        else:
            marginal_dte = dte - prev_dte

            if marginal_dte > 0:
                ret_per_marginal_dte = (
                    (ret - prev_return) / marginal_dte
                )
            else:
                ret_per_marginal_dte = 0

        returns_per_marginal_dte.append(round(ret_per_marginal_dte * 100, 1))

        prev_dte = dte
        prev_return = ret

        hi_delta = calculate_call_delta(
            spot=current_price,
            strike=short_strike,
            dte=dte,
            iv_percent=iv_hi,
            risk_free_rate=RISK_FREE_RATE
        )

        lo_delta = calculate_call_delta(
            spot=current_price,
            strike=long_strike,
            dte=dte,
            iv_percent=iv_lo,
            risk_free_rate=RISK_FREE_RATE
        )

        covered_call_delta = calculate_call_delta(
            spot=current_price,
            strike=covered_call_strike,
            dte=dte,
            iv_percent=iv_cc,
            risk_free_rate=RISK_FREE_RATE
        )

        one_spread_delta = lo_delta - hi_delta
        total_spread_delta = num_spreads * one_spread_delta
        covered_call_delta_contribution = -covered_call_delta

        total_position_delta = (
            1.0
            + covered_call_delta_contribution
            + total_spread_delta
        )

        hi_deltas.append(hi_delta)
        lo_deltas.append(lo_delta)
        covered_call_deltas.append(covered_call_delta)
        spread_delta_per_unit.append(round(one_spread_delta, 5))
        long_call_spread_total_delta.append(round(total_spread_delta, 4))
        covered_call_delta_contributions.append(
            round(covered_call_delta_contribution, 4)
        )
        total_position_deltas.append(round(total_position_delta, 4))

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
        "Return/Marginal DTE": [f"{v}%" for v in returns_per_marginal_dte],
        f"Delta - {short_strike:.0f} call (Hi)": hi_deltas,
        f"Delta - {long_strike:.0f} call (Lo)": lo_deltas,
        f"Delta - {covered_call_strike:.0f} call (Covered Call)": covered_call_deltas,
        "Spread Delta (per single spread)": spread_delta_per_unit,
        "Long Call Spread Delta (total position, x spreads held)": long_call_spread_total_delta,
        "Covered Call Delta Contribution": covered_call_delta_contributions,
        "Equity Delta": [1.0] * len(term_df),
        "Total Position Delta": total_position_deltas
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
        short_strike=short_strike,
        covered_call_strike=covered_call_strike
    )

    expiration_options = [
        f"{row['Expiration']} ({row['DTE']} DTE)"
        for _, row in all_available_expirations.iterrows()
    ]

    # Defaults to showing every available expiration; he can
    # remove any individually using the "x" on each pill.
    default_selection = expiration_options

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

    if covered_call_strike != short_strike:
        st.info(
            f"Note: your covered call strike (${covered_call_strike:.0f}) "
            f"is different from the spread's Hi strike (${short_strike:.0f}). "
            f"The delta section calculates these separately, as it should."
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
        label="📥 Generate & Download Excel Sheet",
        data=excel_bytes,
        file_name=f"BE_call_spread_{long_strike:.0f}_{short_strike:.0f}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"Error: {e}")
