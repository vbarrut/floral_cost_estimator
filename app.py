import os
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# PATHS
# ----------------------------
st.set_page_config(page_title="Floral Cost Estimator", page_icon="💐", layout="centered")

BASE = Path(__file__).parent
DATA = BASE / "data"
MODEL_PATH = BASE / "trained_model.pkl"
PRICING_PATH = DATA / "pricing_config.csv"
STEMS_PATH = DATA / "stem_prices.csv"
STEM_IDX_PATH = DATA / "stem_price_index.txt"
FI_IMG = BASE / "visualizations" / "feature_importance.png"
PVA_IMG = BASE / "visualizations" / "predicted_vs_actual.png"

# ----------------------------
# TAX CALCULATION
# ----------------------------
TAX_DEFAULT = 0.0825

TAXABLE_COMPONENT_KEYS = {"Materials", "Centerpieces"}
TAXABLE_STAFF_KEYS = {"Florals"}  

def compute_tax(amount: float, rate: float = TAX_DEFAULT) -> float:
    try:
        return max(0.0, float(amount) * float(rate))
    except Exception:
        return 0.0

tax_rate = TAX_DEFAULT

# ----------------------------
# SERVICE FEE CALCULATIONS
# ----------------------------
SERVICE_FEE_OPTIONS = {
    "Drop-off only (15%)": 0.15,
    "Drop-off + on-site setup (20%)": 0.20,
    "Drop-off + setup + breakdown (25%)": 0.25,
}

# ----------------------------
# STAFF ACCESS CONTROL (password-protected view)
# ----------------------------
# Try to load staff password from Streamlit Secrets or environment variable
STAFF_PASSWORD = st.secrets.get("FLORAL_APP_STAFF_PASS", "") or os.environ.get("FLORAL_APP_STAFF_PASS", "")

# Initialize session
if "staff_authenticated" not in st.session_state:
    st.session_state.staff_authenticated = False

# Sidebar login prompt
st.sidebar.markdown("### 🔐 Staff Login")

# If no password is configured, lock staff mode
if not STAFF_PASSWORD:
    st.sidebar.warning("Staff password is not configured. Staff mode is currently disabled.")
    password_input = None
    is_staff = False
else:
    # Normal login flow
    password_input = st.sidebar.text_input("Enter staff password", type="password", placeholder="••••••••••")

    if password_input:
        if password_input == STAFF_PASSWORD:
            st.session_state.staff_authenticated = True
            st.sidebar.success("✅ Access granted")
        else:
            st.session_state.staff_authenticated = False
            st.sidebar.error("❌ Invalid password")

    is_staff = st.session_state.staff_authenticated

# ----------------------------
# CACHING
# ----------------------------
@st.cache_data
def load_pricing() -> pd.DataFrame:
    df = pd.read_csv(PRICING_PATH)
    for c in ["materials_unit_baseline","labor_unit_baseline","margin_unit_baseline","price_unit_baseline"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

@st.cache_data
def load_stems() -> pd.DataFrame:
    if STEMS_PATH.exists():
        df = pd.read_csv(STEMS_PATH)
        if "price_per_stem" in df.columns:
            df["price_per_stem"] = pd.to_numeric(df["price_per_stem"], errors="coerce")
        return df
    return pd.DataFrame(columns=["flower_type","price_per_stem"])

@st.cache_resource
def try_load_model():
    # Load model if compatible; otherwise return None so app still runs fine
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def read_stem_index() -> float:
    try:
        return float(Path(STEM_IDX_PATH).read_text().strip())
    except Exception:
        return 1.0

pricing = load_pricing()
stems = load_stems()
model = try_load_model()
baseline_stem_index = read_stem_index()

# ----------------------------
# HELPERS
# ----------------------------
CATEGORY_LABELS = {
    "bridal_bouquet": "Bridal Bouquets",
    #"bridesmaid": "Bridesmaid Bouquets",
    "centerpiece": "Table Centerpieces",
    "ceremony": "Ceremony Flowers",
    "personal": "Wearable Flowers",
    #"bar_decor": "Bar & Welcome Table",
    #"accessory": "Accessories",
    "other": "Other Items",
}

# Friendly descriptions (customer-visible)
STYLE_HINTS = {
    "bridal_bouquet": "Handheld bouquet for the bride (sizes and flower types vary).",
    #"bridesmaid": "Smaller bouquets for attendants, coordinated with the bridal palette.",
    "centerpiece": "Floral arrangements for guest tables (petite to grand).",
    "ceremony": "Statement florals like arches, meadows, clouds, or aisle pieces.",
    "personal": "Wearable flowers (boutonnieres, corsages, hair pieces).",
    #"bar_decor": "Accent florals for bar, welcome table, or signage.",
    #"accessory": "Candles or non-floral accents supporting the floral design.",
    "other": "Additional items not covered above.",
}

def compute_materials_scaler(stems_df: pd.DataFrame, baseline_idx: float) -> float:
    if stems_df.empty:
        return 1.0
    current = stems_df["price_per_stem"].dropna()
    if len(current) == 0 or baseline_idx == 0:
        return 1.0
    return float(current.mean() / baseline_idx)

def subtotal_line(rec: pd.Series, qty: int, materials_scaler: float) -> float:

    mat = float(rec["materials_unit_baseline"]) * materials_scaler
    lab = float(rec["labor_unit_baseline"])
    mar = float(rec["margin_unit_baseline"])
    return (mat + lab + mar) * qty

def format_money(x: float) -> str:
    return f"${x:,.2f}"

# ----------------------------
# SIDEBAR: MODE & BASICS
# ----------------------------
st.sidebar.header("Estimate Setup")
mode = st.sidebar.radio("Who is using the tool?", ["Customer", "Staff (owners)"], index=0)
staff_authed = st.session_state.get("staff_authenticated", False)

is_staff = (mode.startswith("Staff") and staff_authed)

if mode.startswith("Staff") and not staff_authed:
    st.sidebar.warning("Staff mode selected. Enter the staff password in the **Staff Login** section above to unlock.")

st.title("Floral Cost Estimator 💐")
st.caption("Get a quick wedding/event floral estimate. Final pricing is confirmed after a design consultation.")

materials_scaler = compute_materials_scaler(stems, baseline_stem_index)
delivery_mi = st.sidebar.number_input("Delivery distance (miles)", min_value=0.0, max_value=200.0, value=6.0, step=0.5)
delivery_fee = 100.0 + 1.0 * delivery_mi

# Estimated tax controls
st.sidebar.markdown("### Estimated Tax")
if is_staff:
    tax_rate = st.sidebar.number_input("Tax rate (%)", min_value=0.0, max_value=25.0, value=100*TAX_DEFAULT, step=0.25) / 100.0
    #tax_delivery = st.sidebar.checkbox("Apply tax to delivery?", value=False)
    #tax_extras   = st.sidebar.checkbox("Apply tax to extras (staff view)?", value=False)
else:
    tax_rate = TAX_DEFAULT
    tax_delivery = False
    tax_extras = False

st.sidebar.caption(f"Estimated tax rate: {tax_rate*100:.2f}%")

# ----------------------------
# CUSTOMER VIEW
# ----------------------------
if not is_staff:

    ### Step 1 - Event Basics
    with st.expander("Step 1 — Event Basics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            guests = st.number_input("Guest Count (estimate)", min_value=10, max_value=1000, value=120, step=5)
        with col2:
            season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"], index=0)
            service_choice = st.selectbox(
                "Service level",
                list(SERVICE_FEE_OPTIONS.keys()),
                index=0,
                help="Service fee is a percentage of the floral subtotal (pre-tax)."
            )
        st.caption("Tip: You can update these later. They help us calibrate scope and logistics.")

    ### Step 2 - Choose Items
    with st.expander("Step 2 — Choose Items", expanded=True):
        st.write("Pick what you need. Quantities can be changed as you go.")

        chosen_rows = []
        for cat, cat_label in CATEGORY_LABELS.items():
            cat_df = pricing[pricing["category"] == cat]
            if cat_df.empty:
                continue
            with st.container(border=True):
                st.write(f"**{cat_label}**")
                st.caption(STYLE_HINTS.get(cat, ""))


                for _, row in cat_df.iterrows():
                    c1, c2 = st.columns([3,1])
                    with c1:
                        pretty = row["style_label"]
                        st.write(pretty)
                    with c2:
                        qty = st.number_input(
                            label=f"Qty", #– {row['style_key']}",
                            min_value=0, step=1, value=0,
                            key=f"qty_{row['style_key']}"
                        )
                    if qty > 0:
                        chosen_rows.append((row, qty))

    ### Step 3 - Review & Download
    with st.expander("Step 3 — Review & download", expanded=True):

        if chosen_rows:
            items = []
            florals_total = 0.0

            for row, qty in chosen_rows:
                line_total = subtotal_line(row, qty, materials_scaler)
                florals_total += line_total
                items.append({
                    "Item": row["style_label"],
                    "Qty": int(qty),
                    "Estimated Subtotal": round(line_total, 2),
                })

            items_df = pd.DataFrame(items)
            st.markdown("### Your selections")
            st.dataframe(items_df, hide_index=True, use_container_width=True)

            #extras = st.number_input("Optional: rentals or special requests ($)", min_value=0.0, step=10.0, value=0.0)
            #total_est = florals_total + delivery_fee #+ extras

            # Compute service fee from florals subtotal
            service_rate = SERVICE_FEE_OPTIONS.get(service_choice, 0.15)
            service_fee = florals_total * service_rate

            # Estimated tax (customer view)
            total_est = florals_total + service_fee + delivery_fee
            est_tax = compute_tax(total_est, tax_rate)
            grand_total = total_est + est_tax

            # Display breakdown
            st.markdown("### Estimate Summary")
            st.write(f"**Delivery fee:** {format_money(delivery_fee)}")
            st.write(f"**Service fee ({int(service_rate * 100)}% of florals):** {format_money(service_fee)}")

            low, high = grand_total * 0.9, grand_total * 1.1
            # 1st row: main metrics
            colA, colB, colC = st.columns(3)
            colA.metric("Estimated total (pre-tax)", format_money(total_est))
            colB.metric(f"Estimated tax", format_money(est_tax)) #@ {tax_rate * 100:.2f}%
            colC.metric("Grand total (with tax)", format_money(grand_total))

            # 2nd row: ±10% bands (around grand total)
            low_grand = grand_total * 0.90
            high_grand = grand_total * 1.10
            lcol, rcol = st.columns(2)
            lcol.metric("Low estimate (–10%)", format_money(low_grand))
            rcol.metric("High estimate (+10%)", format_money(high_grand))

            st.caption("This is a planning estimate. Final tax and pricing are confirmed in the proposal.")

            # CSV export with full totals
            export_df = items_df.copy()
            export_df.loc[len(export_df)] = ["Delivery", 1, round(delivery_fee, 2)]
            export_df.loc[len(export_df)] = [f"Service fee ({int(service_rate * 100)}% of florals)", "",
                                             round(service_fee, 2)]
            export_df.loc[len(export_df)] = ["Subtotal (pre-tax)", "", round(total_est, 2)]
            export_df.loc[len(export_df)] = ["Estimated tax", "", round(est_tax, 2)]
            export_df.loc[len(export_df)] = ["Grand total (with tax)", "", round(grand_total, 2)]
            export_df.loc[len(export_df)] = ["Low estimate (–10%)", "", round(low_grand, 2)]
            export_df.loc[len(export_df)] = ["High estimate (+10%)", "", round(high_grand, 2)]

            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                "Download estimate (.csv)",
                data=buf.getvalue(),
                file_name="floral_estimate.csv",
                mime="text/csv",
                type="primary"
            )

        else:
            st.info("Select at least one item above to see your estimate.")

    st.markdown("---")
    st.caption(
        "Questions about flowers, colors, or availability? We’ll guide you during the consultation. "
        "Delivery is calculated based on your event location."
    )

# ----------------------------
# STAFF VIEW
# ----------------------------
if is_staff:
    st.success("Staff mode enabled.")
    tabs = st.tabs(["Estimator (detailed)", "Descriptive Analysis", "Model Insights"])

    with tabs[0]:
        st.subheader("Estimator (internal view)")
        st.caption("Materials are scaled by stem price index. Internals are shown for staff only.")

        st.write(
            f"**Stem price index:** current mean = ${stems['price_per_stem'].dropna().mean():,.2f} "
            f"(baseline ${baseline_stem_index:,.2f}) → materials × **{materials_scaler:.2f}**"
        )

        # Quick data editor for staff
        default_rows = pd.DataFrame({
            "style_key": pricing["style_key"].head(6).tolist(),
            "quantity": [1, 5, 0, 0, 0, 0][:len(pricing["style_key"].head(6))]
        })
        grid = st.data_editor(
            default_rows,
            num_rows="dynamic",
            column_config={
                "style_key": st.column_config.SelectboxColumn("Style", options=list(pricing["style_key"]), required=True),
                "quantity": st.column_config.NumberColumn("Qty", min_value=0, step=1),
            },
            use_container_width=True,
            key="staff_editor"
        )

        grid["quantity"] = grid["quantity"].fillna(0).astype(int)
        grid = grid.fillna({"quantity": 0})

        rows = []
        florals_total = 0.0
        for _, r in grid.iterrows():
            sk = r["style_key"]
            qty = int(r.get("quantity") or 0)
            if qty <= 0:
                continue
            rec = pricing.loc[pricing["style_key"] == sk].iloc[0]
            u_mat = float(rec["materials_unit_baseline"]) * materials_scaler
            u_lab = float(rec["labor_unit_baseline"])
            u_mar = float(rec["margin_unit_baseline"])
            u_sum = u_mat + u_lab + u_mar
            line = u_sum * qty
            florals_total += line
            rows.append({
                "Category": rec["category"],
                "Style": rec["style_label"],
                "Qty": qty,
                "Unit Materials ($)": round(u_mat, 2),
                "Unit Labor ($)": round(u_lab, 2),
                "Unit Margin ($)": round(u_mar, 2),
                "Unit Subtotal ($)": round(u_sum, 2),
                "Line Total ($)": round(line, 2),
            })

        if rows:
            df_int = pd.DataFrame(rows)
            st.dataframe(df_int, use_container_width=True, hide_index=True)
        else:
            df_int = pd.DataFrame()
            st.info("Add at least one line to see internals.")

        # Service-level Control
        service_choice_staff = st.selectbox(
            "Service level",
            list(SERVICE_FEE_OPTIONS.keys()),
            index=0,
            help="Applied to the floral subtotal (pre-tax).",
            key="staff_service_choice"
        )
        service_rate_staff = SERVICE_FEE_OPTIONS.get(service_choice_staff, 0.15)
        service_fee_staff = florals_total * service_rate_staff

        # Pre-tax total includes service fee
        base_total = florals_total + service_fee_staff + delivery_fee #+ extras

        col1, col2 = st.columns(2)
        col1.metric("Baseline total", format_money(base_total))

        # Taxable under staff view
        tax_delivery = globals().get("tax_delivery", False)
        tax_extras = globals().get("tax_extras", False)
        tax_service = globals().get("tax_service", True)  # allow service fee to be taxed (common)

        staff_tax_base = florals_total
        if tax_service:
            staff_tax_base += service_fee_staff
        if tax_delivery:
            staff_tax_base += delivery_fee
        if tax_extras:
            staff_tax_base += extras

        est_tax_staff = compute_tax(staff_tax_base, tax_rate)
        grand_total_staff = base_total + est_tax_staff

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline total (pre-tax)", format_money(base_total))
        m2.metric("Estimated tax", format_money(est_tax_staff))
        m3.metric("Grand total (with tax)", format_money(grand_total_staff))

        st.caption(f"Service fee: {int(service_rate_staff * 100)}% of florals → {format_money(service_fee_staff)}")

        # ML comparison with service fee + tax
        if model is not None and not df_int.empty:
            preds = []
            for _, rec in df_int.iterrows():
                pk = pricing[pricing["style_label"] == rec["Style"]].iloc[0]
                feat = pd.DataFrame([{
                    "style_key": pk["style_key"],
                    "quantity": int(rec["Qty"]),
                    "unit_materials": pk["materials_unit_baseline"],
                    "unit_labor": pk["labor_unit_baseline"],
                    "unit_margin": pk["margin_unit_baseline"],
                }])
                preds.append(float(model.predict(feat)[0]))

            ml_total = sum(preds) + service_fee_staff + delivery_fee #+ extras

            ml_tax_base = sum(preds)  # ML proxy for florals
            if tax_service:
                ml_tax_base += service_fee_staff
            if tax_delivery:
                ml_tax_base += delivery_fee
            if tax_extras:
                ml_tax_base += extras

            ml_est_tax = compute_tax(ml_tax_base, tax_rate)
            ml_grand_total = ml_total + ml_est_tax

            col2.metric("ML predicted total (pre-tax)", format_money(ml_total))
            col2.metric("ML estimated tax", format_money(ml_est_tax))
            col2.metric("ML grand total (with tax)", format_money(ml_grand_total))

            st.caption("Note: RF model trained on historical jobs; use for sanity checks, not final pricing.")

        # Composition pies: pre-tax vs with-tax
        parts_pre = {
            "Florals": florals_total,
            "Service Fee": service_fee_staff,
            "Delivery": delivery_fee,
            #"Extras": extras,
        }
        parts_post = dict(parts_pre)
        parts_post["Tax"] = est_tax_staff

        c1, c2 = st.columns(2)
        with c1:
            vals = [v for v in parts_pre.values() if v > 0]
            labels = [k for k, v in parts_pre.items() if v > 0]
            if sum(vals) > 0:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(vals, labels=labels, autopct="%1.0f%%")
                ax.set_title("Composition (pre-tax)")
                st.pyplot(fig)

        with c2:
            vals = [v for v in parts_post.values() if v > 0]
            labels = [k for k, v in parts_post.items() if v > 0]
            if sum(vals) > 0:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(vals, labels=labels, autopct="%1.0f%%")
                ax.set_title("Composition (with tax)")
                st.pyplot(fig)

    with tabs[1]:
        st.subheader("Descriptive Analysis (from sales)")
        show_cols = [
            "style_label", "category", "units_sold",
            "price_unit_baseline", "materials_unit_baseline",
            "labor_unit_baseline", "margin_unit_baseline"
        ]
        if all(c in pricing.columns for c in show_cols):
            st.dataframe(
                pricing[show_cols].rename(columns={
                    "style_label": "Style", "category": "Category", "units_sold": "Units sold",
                    "price_unit_baseline": "Avg unit price", "materials_unit_baseline": "Avg unit materials",
                    "labor_unit_baseline": "Avg unit labor", "margin_unit_baseline": "Avg unit margin",
                }),
                use_container_width=True
            )
        else:
            st.info("Pricing config missing expected columns.")

        st.markdown("**Sample stem prices**")
        st.dataframe(stems.head(25), use_container_width=True)

    with tabs[2]:
        st.subheader("Model Insights")
        if FI_IMG.exists():
            st.image(str(FI_IMG), caption="Top Feature Importances", use_column_width=True)
        if PVA_IMG.exists():
            st.image(str(PVA_IMG), caption="Predicted vs Actual", use_column_width=True)
        if not FI_IMG.exists() and not PVA_IMG.exists():
            st.info("Train the model to generate insights visuals.")
else:
    st.info("Staff mode is locked. Enter the staff password in the **sidebar** to unlock the Staff tabs.")

