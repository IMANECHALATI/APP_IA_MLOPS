import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="Irrigation Intelligente", page_icon="🌱", layout="wide")

# Configuration de l'URL de l'API
API_URL = "http://127.0.0.1:8000/predict"

# ==================== FONCTIONS UTILITAIRES ====================

@st.cache_data
def load_config():
    """Charge et analyse le fichier CSV pour extraire les features et leurs statistiques"""
    paths = ["data/irrigation_prediction.csv", "../data/irrigation_prediction.csv"]
    df = None
    
    for p in paths:
        try:
            df = pd.read_csv(p)
            break
        except:
            continue
    
    if df is None:
        st.error("❌ Fichier CSV introuvable dans `data/` ou `../data/`")
        st.stop()

    # Nettoyage : retirer les colonnes cibles
    target_cols = ["Irrigation", "Status", "Irrigation_Needed"]
    features = df.drop(columns=[c for c in target_cols if c in df.columns], errors='ignore')
    
    # Séparation colonnes catégorielles et numériques
    cat_cols = features.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = features.select_dtypes(include=['number']).columns.tolist()
    
    # Extraction des catégories et statistiques
    categories = {col: sorted(features[col].dropna().unique().tolist()) for col in cat_cols}
    num_stats = {
        col: {
            "min": float(features[col].min()), 
            "max": float(features[col].max()), 
            "mean": float(features[col].mean())
        } for col in num_cols
    }
    
    return categories, num_stats, cat_cols, num_cols


def fetch_weather(city_name):
    """
    Récupère la météo en temps réel via Open-Meteo (API gratuite).
    
    Args:
        city_name: Nom de la ville
    
    Returns:
        dict avec 'temperature', 'humidity', 'city', 'date' ou None si erreur
    """
    try:
        # Étape 1 : Géocodage de la ville
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=fr"
        geo_response = requests.get(geo_url, timeout=5)
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return None
        
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        city_full = geo_data["results"][0]["name"]
        
        # Étape 2 : Récupération météo actuelle
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m"
        )
        weather_response = requests.get(weather_url, timeout=5)
        weather_data = weather_response.json()
        
        return {
            "temperature": round(weather_data["current"]["temperature_2m"], 1),
            "humidity": weather_data["current"]["relative_humidity_2m"],
            "city": city_full,
            "date": datetime.now().strftime("%d/%m/%Y à %H:%M")
        }
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout : l'API météo ne répond pas.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors de la récupération météo : {str(e)}")
        return None


def detect_weather_column(col_name, keyword):
    """Détecte si une colonne correspond à un type de donnée météo (temp, humid, etc.)"""
    col_lower = col_name.lower()
    keywords = keyword if isinstance(keyword, list) else [keyword]
    return any(k in col_lower for k in keywords)


# ==================== INTERFACE PRINCIPALE ====================

st.title("🌱 Système d'Aide à la Décision : Irrigation Intelligente")
st.caption(f"📅 Date du jour : **{datetime.now().strftime('%A %d %B %Y')}**")

# Chargement configuration
categories, num_stats, cat_cols, num_cols = load_config()

# Stockage des données d'entrée
input_data = {}

# ==================== SECTION 1 : MÉTÉO EN TEMPS RÉEL ====================

st.header("🌤️ Données Météorologiques")

use_live_weather = st.toggle(
    "🔴 Activer la météo en temps réel", 
    value=False,
    help="Récupère automatiquement température et humidité depuis Open-Meteo"
)

weather_data = None

if use_live_weather:
    col_city, col_btn = st.columns([3, 1])
    
    with col_city:
        city_input = st.text_input(
            "🏙️ Entrez votre ville", 
            placeholder="Ex: Marrakech, Paris, Casablanca...",
            key="city_input"
        )
    
    with col_btn:
        st.write("")  # Alignement visuel
        fetch_btn = st.button("🔄 Récupérer", use_container_width=True)
    
    # Récupération météo
    if fetch_btn and city_input:
        with st.spinner(f"🌍 Récupération météo pour {city_input}..."):
            weather_data = fetch_weather(city_input)
    
    # Affichage météo récupérée
    if weather_data:
        st.success(f"✅ Météo récupérée pour **{weather_data['city']}** le {weather_data['date']}")
        
        col_temp, col_hum = st.columns(2)
        with col_temp:
            st.metric("🌡️ Température", f"{weather_data['temperature']} °C")
        with col_hum:
            st.metric("💧 Humidité Relative", f"{weather_data['humidity']} %")
    elif fetch_btn:
        st.warning("⚠️ Ville introuvable. Vérifiez l'orthographe ou désactivez le mode temps réel.")

st.divider()

# ==================== SECTION 2 : PARAMÈTRES DE CULTURE ====================

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📋 Informations sur la Culture")
    for col in cat_cols:
        label = col.replace('_', ' ').title()
        input_data[col] = st.selectbox(
            f"🌾 {label}", 
            categories[col],
            key=f"cat_{col}"
        )

with col_right:
    st.subheader("🌍 Conditions Environnementales")
    
    # Gestion intelligente : météo live remplace temp/humid
    for col in num_cols:
        # Identifier le type de donnée
        is_temp = detect_weather_column(col, ["temp", "temperature"])
        is_humid = detect_weather_column(col, ["humid", "moisture"])
        is_weather_related = detect_weather_column(col, ["rain", "wind", "sun", "precipitation"])
        
        # Déterminer si c'est une donnée météo prioritaire
        is_priority_weather = is_temp or is_humid or is_weather_related
        
        if is_priority_weather:
            # Si météo live activée et données dispo
            if use_live_weather and weather_data:
                if is_temp:
                    input_data[col] = weather_data["temperature"]
                    st.metric(f"🌡️ {col.replace('_', ' ').title()}", f"{weather_data['temperature']} °C", delta="Temps réel")
                elif is_humid:
                    input_data[col] = weather_data["humidity"]
                    st.metric(f"💧 {col.replace('_', ' ').title()}", f"{weather_data['humidity']} %", delta="Temps réel")
                else:
                    # Autres données météo (pluie, vent...) restent manuelles
                    input_data[col] = st.slider(
                        f"🌦️ {col.replace('_', ' ').title()}", 
                        num_stats[col]["min"], 
                        num_stats[col]["max"], 
                        num_stats[col]["mean"],
                        key=f"num_{col}"
                    )
            else:
                # Mode manuel pour toutes les données météo
                input_data[col] = st.slider(
                    f"🌦️ {col.replace('_', ' ').title()}", 
                    num_stats[col]["min"], 
                    num_stats[col]["max"], 
                    num_stats[col]["mean"],
                    key=f"num_{col}"
                )

st.divider()

# ==================== SECTION 3 : PARAMÈTRES AVANCÉS (OPTIONNELS) ====================

with st.expander("🔬 Paramètres Avancés du Sol (pH, Carbone, Azote, etc.)", expanded=False):
    st.caption("Ces paramètres affinent la précision mais peuvent rester aux valeurs par défaut.")
    
    advanced_cols = st.columns(2)
    col_idx = 0
    
    for col in num_cols:
        if col not in input_data:  # Uniquement les colonnes non déjà remplies
            with advanced_cols[col_idx % 2]:
                input_data[col] = st.slider(
                    f"⚗️ {col.replace('_', ' ').title()}", 
                    num_stats[col]["min"], 
                    num_stats[col]["max"], 
                    num_stats[col]["mean"],
                    key=f"adv_{col}"
                )
            col_idx += 1

st.divider()

# ==================== MAPPING DES PRÉDICTIONS ====================

# Dictionnaire pour convertir les prédictions en labels lisibles
IRRIGATION_LABELS = {
    0: "❌ Pas d'irrigation nécessaire",
    1: "💧 Irrigation modérée recommandée",
    2: "💦 Irrigation forte nécessaire",
    "0": "❌ Pas d'irrigation nécessaire",
    "1": "💧 Irrigation modérée recommandée", 
    "2": "💦 Irrigation forte nécessaire",
    "Low": "❌ Irrigation faible",
    "Medium": "💧 Irrigation modérée",
    "High": "💦 Irrigation forte",
    "No": "❌ Pas d'irrigation",
    "Yes": "💦 Irrigation nécessaire"
}

# ==================== SECTION 4 : PRÉDICTION ====================

st.subheader("🚀 Lancer l'Analyse")

if st.button("💧 PRÉDIRE LE BESOIN EN IRRIGATION", type="primary", use_container_width=True):
    with st.spinner("🧠 Le modèle analyse vos données..."):
        try:
            # Appel API FastAPI
            response = requests.post(API_URL, json=input_data, timeout=10)
            
            if response.status_code == 200:
                prediction = response.json()
                
                # Extraction robuste du résultat
                raw_result = (
                    prediction.get("prediction") or 
                    prediction.get("result") or 
                    prediction.get("status") or 
                    list(prediction.values())[0]
                )
                
                # Conversion en label lisible
                final_result = IRRIGATION_LABELS.get(raw_result, f"Code : {raw_result}")
                
                # Affichage du résultat avec style selon le niveau
                st.balloons()
                
                # Style conditionnel selon le résultat
                if raw_result in [0, "0", "No", "Low"]:
                    st.success(f"### {final_result}")
                    st.info("💡 Le sol a suffisamment d'humidité. Économisez l'eau !")
                elif raw_result in [1, "1", "Medium"]:
                    st.warning(f"### {final_result}")
                    st.info("💡 Une irrigation légère est conseillée.")
                else:  # High, 2, "2", Yes
                    st.error(f"### {final_result}")
                    st.info("💡 Irrigation urgente recommandée pour optimiser la culture.")
                
                # Affichage des données envoyées (debug)
                with st.expander("📊 Voir les données envoyées au modèle"):
                    st.json(input_data)
                    st.caption(f"Prédiction brute du modèle : **{raw_result}**")
            
            else:
                st.error(f"❌ Erreur API (Code {response.status_code}). Vérifiez votre backend FastAPI.")
                st.code(response.text)
        
        except requests.exceptions.ConnectionError:
            st.error("🔌 **Impossible de se connecter à l'API FastAPI.**")
            st.info("💡 Assurez-vous que FastAPI tourne : `uvicorn src.api:app --reload`")
        
        except requests.exceptions.Timeout:
            st.error("⏱️ L'API a mis trop de temps à répondre (timeout).")
        
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {str(e)}")

# ==================== FOOTER ====================

st.divider()
st.caption("🌱 Système MLOps d'Irrigation Intelligente | Développé avec Streamlit + FastAPI + ML")