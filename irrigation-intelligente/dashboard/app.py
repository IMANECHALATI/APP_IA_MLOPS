import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px

# ──────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris - Irrigation Intelligente",
    page_icon="🌿",
    layout="wide"
)

# Style CSS pour le look "Luxe & Pastel"
st.markdown("""
    <style>
    .main {
        background-color: #F8FAF8;
    }
    .stButton>button {
        background-color: #A3B18A;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 25px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #588157;
        transform: scale(1.05);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #A3B18A;
    }
    h1, h2, h3 {
        color: #3A5A40;
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# BARRE LATÉRALE (NAVIGATION)
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=100)
    st.title("IRIS Project")
    st.markdown("---")
    menu = st.radio("Navigation", ["Tableau de Bord", "Nouvelle Prédiction", "Santé du Système"])
    
    st.info("Système connecté à l'API MLOps v1.0.0")

# ──────────────────────────────────────────────────────────
# LOGIQUE DE PRÉDICTION
# ──────────────────────────────────────────────────────────
if menu == "Nouvelle Prédiction":
    st.header("🌿 Analyse des Besoins en Irrigation")
    st.subheader("Entrez les paramètres agronomiques")

    # Organisation en colonnes pour la beauté du formulaire
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌍 Sol & Localisation")
        soil_type = st.selectbox("Type de Sol", ["Sandy", "Loamy", "Clay", "Silty"])
        region = st.selectbox("Région", ["North", "South", "East", "West"])
        field_area = st.number_input("Surface (Hectare)", min_value=0.1, value=2.5)
        soil_ph = st.slider("pH du Sol", 0.0, 14.0, 6.5)

    with col2:
        st.markdown("### 🌾 Culture & Stade")
        crop_type = st.selectbox("Type de Culture", ["Wheat", "Maize", "Rice", "Cotton"])
        growth_stage = st.selectbox("Stade de Croissance", ["Initial", "Vegetative", "Flowering", "Maturity"])
        mulching = st.radio("Mulching utilisé ?", ["Yes", "No"])
        season = st.selectbox("Saison", ["Summer", "Winter", "Spring", "Autumn"])

    with col3:
        st.markdown("### ☁️ Météo & Eau")
        temp = st.number_input("Température (°C)", value=25.0)
        humidity = st.slider("Humidité (%)", 0, 100, 50)
        rainfall = st.number_input("Précipitations (mm)", value=0.0)
        moisture = st.number_input("Humidité du Sol (%)", value=30.0)

    st.markdown("---")
    
    if st.button("Lancer l'Analyse"):
        # Préparation des données pour l'API
        payload = {
            "Soil_Type": soil_type, "Soil_pH": soil_ph, "Soil_Moisture": moisture,
            "Organic_Carbon": 1.2, "Electrical_Conductivity": 0.5, "Temperature_C": temp,
            "Humidity": humidity, "Rainfall_mm": rainfall, "Sunlight_Hours": 8.0,
            "Wind_Speed_kmh": 12.0, "Crop_Type": crop_type, "Crop_Growth_Stage": growth_stage,
            "Season": season, "Irrigation_Type": "Drip", "Water_Source": "Groundwater",
            "Field_Area_hectare": field_area, "Mulching_Used": mulching,
            "Previous_Irrigation_mm": 10.0, "Region": region
        }

        try:
            with st.spinner('L’IA analyse vos champs...'):
                response = requests.post("http://localhost:8000/predict", json=payload)
                res = response.json()

            if response.status_code == 200:
                st.balloons()
                st.success("Analyse terminée avec succès !")
                
                # Affichage luxueux du résultat
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>Résultat de l'IA</h3>
                        <h1 style='color: #588157;'>{res['irrigation_need']}</h1>
                        <p>Confiance du modèle : <b>{res['confidence']*100:.2f}%</b></p>
                        <p><small>Généré le : {res['timestamp']}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with res_col2:
                    # Petit gauge chart simple
                    fig = px.pie(values=[res['confidence'], 1-res['confidence']], 
                                names=['Confiance', 'Incertitude'],
                                color_discrete_sequence=['#A3B18A', '#DAD7CD'],
                                hole=0.7)
                    fig.update_layout(showlegend=False, height=200, margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig)
            else:
                st.error(f"Erreur API : {res.get('detail')}")
        except Exception as e:
            st.error(f"Impossible de contacter l'API. Vérifiez qu'elle tourne sur le port 8000. Erreur: {e}")

# ──────────────────────────────────────────────────────────
# TABLEAU DE BORD (VISUALISATION)
# ──────────────────────────────────────────────────────────
elif menu == "Tableau de Bord":
    st.header("📊 Vue d'ensemble de l'Exploitation")
    
    # Métriques clés en haut
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Température Moyenne", "24°C", "+2°C")
    m2.metric("Humidité Moyenne", "62%", "-5%")
    m3.metric("Consommation d'eau", "1200 m³", "Eco-mode")
    m4.metric("Santé des cultures", "98%", "Optimale")

    # Graphique fictif pour le style
    st.markdown("### Évolution de l'humidité du sol")
    chart_data = pd.DataFrame({
        'Jour': ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'],
        'Humidité': [30, 35, 28, 45, 40, 38, 33]
    })
    fig = px.line(chart_data, x='Jour', y='Humidité', markers=True,
                  color_discrete_sequence=['#A3B18A'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────
# SANTÉ DU SYSTÈME
# ──────────────────────────────────────────────────────────
elif menu == "Santé du Système":
    st.header("🛠️ Monitoring MLOps")
    try:
        health_res = requests.get("http://localhost:8000/health").json()
        metrics_res = requests.get("http://localhost:8000/metrics").json()
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("### État Serveur")
            st.json(health_res)
        with c2:
            st.write("### Performance Modèle")
            st.write(f"Taux de succès : **{metrics_res['success_rate']*100:.1f}%**")
            st.progress(metrics_res['success_rate'])
    except:
        st.warning("L'API est déconnectée. Les métriques ne sont pas disponibles.")