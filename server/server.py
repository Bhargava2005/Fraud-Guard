
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/get_states', methods=['GET'])
def get_states():
    try:
        df_states     = pd.read_csv('./assets/geo_analysis_updated.csv', usecols=['state'])
        unique_states = sorted(df_states['state'].dropna().unique().tolist())
        return jsonify(unique_states)
    except Exception as e:
        print(f"Error fetching states: {e}")
        return jsonify({"error": "Could not retrieve states"}), 500
    
@app.route('/get_geo_lists', methods=['GET'])
def get_geo_lists():
    state_name = request.args.get('state', '').strip()

    if not state_name or state_name == "All India":
        return jsonify({"zones": [], "districts": []})

    try:
        df = pd.read_csv('./assets/geo_analysis_updated.csv', usecols=['state', 'zone', 'district'])
        state_df = df[df['state'].str.lower() == state_name.lower()]

        # ── FIX 2: only include zones if they actually exist (not all NaN) ──
        raw_zones = state_df['zone'].dropna().unique().tolist()
        zones     = sorted([z for z in raw_zones if str(z).strip() not in ("", "nan", "None")])

        districts = sorted(state_df['district'].dropna().unique().tolist())

        return jsonify({
            "state":     state_name,
            "zones":     zones,       # empty list if all NaN → frontend disables zone dropdown
            "districts": districts,
        })

    except Exception as e:
        print(f"Error retrieving geo lists: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/get_pincodes', methods=['GET'])
def get_pincodes_categorized():
    print("pins endpoint calling")
    try:

        state    = request.args.get('state',    '')
        zone     = request.args.get('zone',     '')
        district = request.args.get('district', '')

        wanted_types = request.args.get('types', 'high,moderate,safe').split(',')

        p1 = float(request.args.get('point1', 30))
        p2 = float(request.args.get('point2', 60))

        df = pd.read_csv('./assets/geo_analysis_updated.csv')
        print("Total rows:", len(df))

        # ── Geographical Filters ──
        if state:
            df = df[df['state'].str.lower() == state.lower()]

        if zone:
            zone_col_valid = df['zone'].notna().any()
            if zone_col_valid:
                zone_matches = df['zone'].str.lower() == zone.lower()
                if zone_matches.any():
                    df = df[zone_matches]
                else:
                    print(f"Zone '{zone}' not found or all NaN — showing all state pincodes")
            else:
                print("Zone column entirely NaN for this state — skipping zone filter")

        if district:
            df = df[df['district'].str.lower() == district.lower()]

        print("Filtered rows:", len(df))

        # ── FIX: Drop rows where latitude or longitude is NaN/missing BEFORE iterating ──
        before = len(df)
        df = df.dropna(subset=['latitude', 'longitude'])

        # Also drop rows where lat/lng are not valid finite numbers
        df = df[
            pd.to_numeric(df['latitude'],  errors='coerce').notna() &
            pd.to_numeric(df['longitude'], errors='coerce').notna()
        ]

        # Convert once to float and filter out non-finite values (inf, -inf)
        df['latitude']  = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df = df[
            df['latitude'].apply(np.isfinite) &
            df['longitude'].apply(np.isfinite)
        ]

        # Also enforce valid coordinate ranges
        df = df[
            (df['latitude']  >= -90)  & (df['latitude']  <= 90)  &
            (df['longitude'] >= -180) & (df['longitude'] <= 180)
        ]

        after = len(df)
        if before != after:
            print(f"⚠️ Dropped {before - after} rows with invalid lat/lng coordinates")

        result = {t: [] for t in wanted_types}

        for _, row in df.iterrows():
            try:
                risk_val = float(
                    (row['high_risk_count_1'] / row['total_customers']) * 100
                ) if row['total_customers'] > 0 else 0.0
            except:
                risk_val = 0.0

            if risk_val < p1:
                category = "safe"
            elif p1 <= risk_val < p2:
                category = "moderate"
            else:
                category = "high"

            if category in result:
                zone_val = row.get('zone', None)
                zone_str = str(zone_val) if zone_val and not (
                    isinstance(zone_val, float) and np.isnan(zone_val)
                ) else "N/A"

                result[category].append({
                    "pincode":         str(row['pin_code']),
                    "lat":             float(row['latitude']),
                    "lang":            float(row['longitude']),
                    "st":              str(row['state']),
                    "dv":              zone_str,
                    "dist":            str(row['district']),
                    "risk_percent":    round(risk_val, 2),
                    "category":        category,
                    "total_customers": int(row['total_customers']),
                })

        return jsonify(result)

    except Exception as e:
        print(f"Error in categorization: {e}")
        return jsonify({"error": str(e)}), 500
    
# =========================
# HELPER FUNCTIONS
# =========================

def not_found_response(entity, id_field, id_value):
    return jsonify({
        "error"    : f"{entity} not found",
        "id_field" : f"{id_field}",
        "id_value" : f"{id_value}"
    }), 404

def safe_encode(value, mapping, default=0):
    """Encode a string category to integer safely."""
    return mapping.get(str(value).strip(), default)

def compute_risk_scores(order_id):

    if not order_id:
        return {"error": "order_id is required"}, 400
    try:
        order_id = int(order_id)
    except ValueError:
        return {"error": "order_id must be an integer"}, 400

    # ═══════════════════════════════════════════════════════════════
    # 1. LOAD ALL MODELS + SCALERS
    # ═══════════════════════════════════════════════════════════════

    customer_model   = joblib.load("./trained_models/customer_risk_models/customer_risk_model.pkl")
    customer_scaler  = joblib.load("./trained_models/customer_risk_models/customer_risk_scaler.pkl")

    device_model     = joblib.load("./trained_models/device_risk_models/device_risk_model.pkl")
    device_scaler    = joblib.load("./trained_models/device_risk_models/device_risk_scaler.pkl")
    device_iso       = joblib.load("./trained_models/device_risk_models/device_risk_isolation.pkl")

    logistics_model  = joblib.load("./trained_models/logistic_risk_models/logistics_risk_model.pkl")
    logistics_scaler = joblib.load("./trained_models/logistic_risk_models/logistics_risk_scaler.pkl")

    order_model      = joblib.load("./trained_models/order_risk_models/order_risk_model.pkl")
    order_scaler     = joblib.load("./trained_models/order_risk_models/order_risk_scaler.pkl")

    product_model    = joblib.load("./trained_models/product_risk_models/product_risk_model.pkl")
    product_scaler   = joblib.load("./trained_models/product_risk_models/product_risk_scaler.pkl")

    seller_model     = joblib.load("./trained_models/seller_risk_models/seller_risk_model.pkl")
    seller_scaler    = joblib.load("./trained_models/seller_risk_models/seller_risk_scaler.pkl")

    # ═══════════════════════════════════════════════════════════════
    # 2. LOAD CSV DATA
    # ═══════════════════════════════════════════════════════════════
    customer_df  = pd.read_csv("./sample_database/customer_data.csv")
    device_df    = pd.read_csv("./sample_database/device_data.csv")
    logistics_df = pd.read_csv("./sample_database/logistics_data.csv")
    order_df     = pd.read_csv("./sample_database/order_data.csv")
    product_df   = pd.read_csv("./sample_database/product_data.csv")
    seller_df    = pd.read_csv("./sample_database/seller_data.csv")

    # ═══════════════════════════════════════════════════════════════
    # 3. FETCH ROWS
    # ═══════════════════════════════════════════════════════════════
    order_row = order_df[order_df['order_id'] == order_id]
    if order_row.empty:
        return not_found_response("Order", "order_id", order_id)
    order_row = order_row.iloc[0]

    customer_id  = order_row['customer_id']
    customer_row = customer_df[customer_df['customer_id'] == customer_id]
    if customer_row.empty:
        return not_found_response("Customer", "customer_id", customer_id)
    customer_row = customer_row.iloc[0]

    device_rows = device_df[device_df['customer_id'] == customer_id]
    if device_rows.empty:
        return not_found_response("Device", "customer_id", customer_id)
    device_row = device_rows.iloc[-1]

    logistics_row = logistics_df[logistics_df['order_id'] == order_id]
    if logistics_row.empty:
        return not_found_response("Logistics", "order_id", order_id)
    logistics_row = logistics_row.iloc[0]

    product_id  = order_row['product_id']
    product_row = product_df[product_df['product_id'] == product_id]
    if product_row.empty:
        return not_found_response("Product", "product_id", product_id)
    product_row = product_row.iloc[0]

    seller_id  = product_row['seller_id']
    seller_row = seller_df[seller_df['seller_id'] == seller_id]
    if seller_row.empty:
        return not_found_response("Seller", "seller_id", seller_id)
    seller_row = seller_row.iloc[0]

    # ═══════════════════════════════════════════════════════════════
    # 4. BUILD INPUTS & PREDICT
    # ═══════════════════════════════════════════════════════════════

    # ── CUSTOMER ─────────────────────────────────────────────────
    customer_input = np.array([[
        float(customer_row['account_age_days']),
        float(customer_row['total_orders']),
        float(customer_row['return_rate']),
        float(customer_row['avg_order_value']),
        float(customer_row['cod_percentage']),
        float(customer_row['end_window_returns']),
        float(customer_row['damaged_claim_rate']),
        float(customer_row['linked_accounts']),
        float(customer_row['vpn_usage']),
        float(customer_row['past_fraud_flag']),
    ]], dtype=float)
    customer_input = customer_scaler.transform(customer_input)
    customer_risk  = float(customer_model.predict_proba(customer_input)[0][1])

    # ── DEVICE ───────────────────────────────────────────────────
    # ⚠️ CRITICAL PIPELINE ORDER (must match training exactly):
    #
    # During TRAINING the order was:
    #   Step 1 → X = raw 10 features
    #   Step 2 → X['anomaly_score'] = iso.fit_predict(X)   ← anomaly appended FIRST
    #   Step 3 → X_scaled = scaler.fit_transform(X)        ← scaler sees 11 features
    #
    # So during PREDICTION the order must be:
    #   Step 1 → build raw 10 features
    #   Step 2 → iso.predict(raw_10) → append → 11 features  ← anomaly FIRST
    #   Step 3 → scaler.transform(11 features)               ← scale AFTER
    #   Step 4 → model.predict_proba(scaled_11)
    #
    # Previous version had scaler before iso → ValueError: expects 11, got 10

    DEVICE_TYPE_MAP = {'Mobile': 0, 'Desktop': 2, 'Laptop': 1}
    device_type_encoded = safe_encode(device_row['device_type'], DEVICE_TYPE_MAP)

    device_raw = np.array([[
        float(device_row['device_age_days']),
        float(device_type_encoded),
        float(device_row['accounts_per_device']),
        float(device_row['emulator_detected']),
        float(device_row['rooted_or_jailbroken']),
        float(device_row['vpn_used']),
        float(device_row['ip_changes_24h']),
        float(device_row['geo_distance_km']),
        float(device_row['login_frequency']),
        float(device_row['failed_login_attempts']),
    ]], dtype=float)                                          # shape: (1, 10)

    # Step 2: append anomaly score → 11 features
    anomaly_score = device_iso.predict(device_raw)            # uses raw 10 features
    device_11     = np.append(device_raw,
                              anomaly_score.reshape(-1, 1),
                              axis=1)                         # shape: (1, 11)

    # Step 3: scale all 11 features together
    device_input  = device_scaler.transform(device_11)        # scaler expects 11
    device_risk   = float(device_model.predict_proba(device_input)[0][1])

    # ── LOGISTICS ────────────────────────────────────────────────
    logistics_input = np.array([[
        float(logistics_row['courier_risk_score']),
        float(logistics_row['delivery_attempts']),
        float(logistics_row['delivery_delay_days']),
        float(logistics_row['otp_confirmation']),
        float(logistics_row['delivery_photo']),
        float(logistics_row['pickup_delay_days']),
        float(logistics_row['pickup_attempts']),
        float(logistics_row['tamper_route']),
        float(logistics_row['distance_km']),
        float(logistics_row['weight_mismatch']),
    ]], dtype=float)
    logistics_input = logistics_scaler.transform(logistics_input)
    logistics_risk  = float(logistics_model.predict_proba(logistics_input)[0][1])

    # ── ORDER ────────────────────────────────────────────────────
    PAYMENT_METHOD_MAP = {'COD': 1, 'Net Banking':0}
    payment_encoded = safe_encode(order_row['payment_method'], PAYMENT_METHOD_MAP)

    order_input = np.array([[
        float(order_row['order_value']),
        float(order_row['item_quantity']),
        float(order_row['identical_items']),
        float(payment_encoded),
        float(order_row['failed_payment_attempts']),
        float(order_row['order_hour']),
        float(order_row['checkout_time_sec']),
        float(order_row['address_mismatch']),
        float(order_row['order_velocity']),
        float(order_row['customer_tenure_days']),
    ]], dtype=float)
    order_input = order_scaler.transform(order_input)
    order_risk  = float(order_model.predict_proba(order_input)[0][1])

    # ── PRODUCT ──────────────────────────────────────────────────
    PRODUCT_CATEGORY_MAP = {
        'Electronics': 0, 'Automotive':0,
        'Fashion'    : 2, 'Sports'  : 1, 'Grocery' : 2,
        'Books'   : 1, 'Toys'    : 1
    }
    category_encoded = safe_encode(product_row['product_category'], PRODUCT_CATEGORY_MAP)

    product_input = np.array([[
        float(product_row['product_price']),
        float(category_encoded),
        float(product_row['return_rate']),
        float(product_row['fraud_return_rate']),
        float(product_row['avg_days_to_return']),
        float(product_row['serial_tracked']),
        float(product_row['counterfeit_risk']),
        float(product_row['fragile']),
        float(product_row['seller_product_risk']),
        float(product_row['discount_percentage']),
    ]], dtype=float)
    product_input = product_scaler.transform(product_input)
    product_risk  = float(product_model.predict_proba(product_input)[0][1])

    # ── SELLER ───────────────────────────────────────────────────
    VERIFICATION_MAP = {'Unverified': 0, 'Basic': 1, 'Partial': 1, 'Premium': 2, 'Full': 2}
    verification_encoded = safe_encode(seller_row['verification_level'], VERIFICATION_MAP)

    seller_input = np.array([[
        float(seller_row['seller_age_days']),
        float(verification_encoded),
        float(seller_row['total_orders']),
        float(seller_row['seller_return_rate']),
        float(seller_row['seller_dispute_rate']),
        float(seller_row['wrong_item_rate']),
        float(seller_row['damaged_item_rate']),
        float(seller_row['avg_product_price']),
        float(seller_row['refund_before_inspection']),
        float(seller_row['negative_feedback_rate']),
    ]], dtype=float)
    seller_input = seller_scaler.transform(seller_input)
    seller_risk  = float(seller_model.predict_proba(seller_input)[0][1])

    # ═══════════════════════════════════════════════════════════════
    # 5. COMPUTE FINAL SCORE
    # ═══════════════════════════════════════════════════════════════
    final_score = float(np.mean([
        customer_risk,
        device_risk,
        logistics_risk,
        order_risk,
        product_risk,
        seller_risk,
    ]))

    # ═══════════════════════════════════════════════════════════════
    # 6. BUILD RESPONSE
    # ═══════════════════════════════════════════════════════════════
    result = {
        "order_id"               : order_id,
        "customer_id"            : int(customer_id),
        "product_id"             : int(product_id),
        "seller_id"              : int(seller_id),
        "risk_scores"            : {
            "customer_risk"      : round(customer_risk   * 100, 2),
            "device_risk"        : round(device_risk     * 100, 2),
            "logistics_risk"     : round(logistics_risk  * 100, 2),
            "order_risk"         : round(order_risk      * 100, 2),
            "product_risk"       : round(product_risk    * 100, 2),
            "seller_risk"        : round(seller_risk     * 100, 2),
        },
        "final_fraud_probability": round(final_score * 100, 2),
        "fraud_alert"            : final_score >= 0.5,
        "risk_level"             : (
            "CRITICAL" if final_score >= 0.75 else
            "HIGH"     if final_score >= 0.50 else
            "MEDIUM"   if final_score >= 0.25 else
            "LOW"
        ),
    }

    print("\n" + "="*55)
    print(f"  FRAUD RISK REPORT — Order #{order_id}")
    print("="*55)
    print(f"  Customer Risk     : {result['risk_scores']['customer_risk']:>6.2f}%")
    print(f"  Device Risk       : {result['risk_scores']['device_risk']:>6.2f}%")
    print(f"  Logistics Risk    : {result['risk_scores']['logistics_risk']:>6.2f}%")
    print(f"  Order Risk        : {result['risk_scores']['order_risk']:>6.2f}%")
    print(f"  Product Risk      : {result['risk_scores']['product_risk']:>6.2f}%")
    print(f"  Seller Risk       : {result['risk_scores']['seller_risk']:>6.2f}%")
    print("-"*55)
    print(f"  Final Fraud Chance: {result['final_fraud_probability']:>6.2f}%")
    print(f"  Risk Level        : {result['risk_level']}")
    print(f"  Fraud Alert       : {'🔴 YES' if result['fraud_alert'] else '🟢 NO'}")
    print("="*55 + "\n")

    return jsonify(result)




@app.route('/predict/fraud-risk', methods=['GET'])
def predict_fraud_risk():

    # ── 1. GET order_id FROM REQUEST ──────────────────────────
    order_id = request.args.get('order_id')
    return compute_risk_scores(order_id)
if __name__ == '__main__':
    app.run(debug=True, port=5000)