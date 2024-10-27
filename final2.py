import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Charger le dataset
df = pd.read_csv('creditdata_cleaned.csv')

# Charger le modèle et le scaler
model = joblib.load('model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')

# Charger les encodeurs pour les colonnes catégorielles
encoders = {}
categorical_columns = ['duree_pret', 'note_emprunteur', 'sous_note_emprunteur', 'anciennete_emploi',
                       'statut_logement', 'statut_verification_revenu', 'objet_pret', 'type_demande']

for column in categorical_columns:
    encoders[column] = joblib.load(f'{column}_encoder.pkl')

# Fonction pour prédire la solvabilité
def predict_solvability(data):
    # Standardiser les colonnes numériques
    numeric_columns = ['montant_pret', 'revenu_annuel', 'taux_endettement']
    data[numeric_columns] = scaler.transform(data[numeric_columns])
    
    # Encoder les colonnes catégorielles
    for column in categorical_columns:
        data[column] = encoders[column].transform(data[column])
    
    # Faire la prédiction
    prediction = model.predict(data)
    return prediction

# Charger les modèles pour la prédiction du montant et de la durée
xgb_model_credit = joblib.load('xgb_model_credit.pkl')
xgb_model_term = joblib.load('xgb_model_term.pkl')
scaler_credit_xgboost = joblib.load('scaler_credit_xgboost')

# Interface utilisateur Streamlit
st.set_page_config(page_title="Application de Prédiction de Crédit", layout="wide")
st.title("Application de Prédiction de Crédit")

# CSS pour le style
st.markdown("""
    <style>
    .bordered {
        border: 2px solid #0072B1;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    h2 {
        color: red;
        font-family: 'Arial', sans-serif;
        font-size: 24px; /* Taille de police des titres */
        text-align: center; /* Centrer les titres */
    }
    label {
        color: red; /* Couleur rouge pour les étiquettes */
        font-weight: bold;
        font-family: 'Arial', sans-serif;
        font-size: 20px; /* Taille de police agrandie pour les étiquettes */
    }
    .result {
        font-size: 24px; /* Taille de police agrandie pour les résultats */
        color: green;
        font-weight: bold;
    }
    .error {
        font-size: 24px; /* Taille de police agrandie pour les erreurs */
        color: red;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Créer deux sections pour les prédictions
with st.container():
    # Écran pour prédire la solvabilité
    with st.expander("Prédiction de la Solvabilité", expanded=True):
        montant_pret = st.number_input("Montant du prêt", min_value=0.0)
        montant_finance = st.number_input("Montant financé", min_value=0.0)
        duree_pret = st.selectbox("Durée du prêt", options=df['duree_pret'].unique())
        taux_interet = st.number_input("Taux d'intérêt (%)", min_value=0.0)
        mensualite = st.number_input("Mensualité", min_value=0.0)
        note_emprunteur = st.selectbox("Note de l'emprunteur", options=df['note_emprunteur'].unique())
        sous_note_emprunteur = st.selectbox("Sous-note de l'emprunteur", options=df['sous_note_emprunteur'].unique())
        anciennete_emploi = st.selectbox("Ancienneté dans l'emploi", options=df['anciennete_emploi'].unique())
        statut_logement = st.selectbox("Statut du logement", options=df['statut_logement'].unique())
        revenu_annuel = st.number_input("Revenu annuel", min_value=0.0)
        statut_verification_revenu = st.selectbox("Statut de la vérification des revenus", options=df['statut_verification_revenu'].unique())
        objet_pret = st.selectbox("Objet du prêt", options=df['objet_pret'].unique())
        taux_endettement = st.number_input("Taux d'endettement (%)", min_value=0.0)
        demandes_credit_6_mois = st.number_input("Demandes de crédit (6 mois)", min_value=0)
        comptes_credit_ouverts = st.number_input("Comptes de crédit ouverts", min_value=0)
        dossiers_publics = st.number_input("Dossiers publics", min_value=0)
        solde_revolving = st.number_input("Solde revolving", min_value=0.0)
        taux_utilisation_revolving = st.number_input("Taux d'utilisation revolving (%)", min_value=0.0)
        comptes_credit_total = st.number_input("Total des comptes de crédit", min_value=0)
        creances_12_mois = st.number_input("Créances des 12 derniers mois", min_value=0)
        type_demande = st.selectbox("Type de demande", options=df['type_demande'].unique())

        # Créer un DataFrame avec les entrées
        data = pd.DataFrame({
            'montant_pret': [montant_pret],
            'montant_finance': [montant_finance],
            'duree_pret': [duree_pret],
            'taux_interet': [taux_interet],
            'mensualite': [mensualite],
            'note_emprunteur': [note_emprunteur],
            'sous_note_emprunteur': [sous_note_emprunteur],
            'anciennete_emploi': [anciennete_emploi],
            'statut_logement': [statut_logement],
            'revenu_annuel': [revenu_annuel],
            'statut_verification_revenu': [statut_verification_revenu],
            'objet_pret': [objet_pret],
            'taux_endettement': [taux_endettement],
            'demandes_credit_6_mois': [demandes_credit_6_mois],
            'comptes_credit_ouverts': [comptes_credit_ouverts],
            'dossiers_publics': [dossiers_publics],
            'solde_revolving': [solde_revolving],
            'taux_utilisation_revolving': [taux_utilisation_revolving],
            'comptes_credit_total': [comptes_credit_total],
            'creances_12_mois': [creances_12_mois],
            'type_demande': [type_demande]
        })

        # Bouton pour effectuer la prédiction
        if st.button("Prédire la solvabilité"):
            prediction = predict_solvability(data)
            if prediction == 1:
                st.markdown('<p class="result">L\'emprunteur est solvable.</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="error">L\'emprunteur n\'est pas solvable.</p>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Écran pour prédire le montant et la durée du prêt
    with st.expander("Prédiction du Montant de Prêt et de Durée", expanded=True):
        montant_total_paye = st.number_input("Montant total payé (FCFA)", min_value=0.0)
        montant_principal_rembourse = st.number_input("Montant principal remboursé (FCFA)", min_value=0.0)
        montant_interets_rembourses = st.number_input("Montant des intérêts remboursés (FCFA)", min_value=0.0)
        montant_dernier_paiement = st.number_input("Montant du dernier paiement (FCFA)", min_value=0.0)
        montant_recouvrement = st.number_input("Montant de recouvrement (FCFA)", min_value=0.0)
        montant_principal_restant = st.number_input("Montant principal restant (FCFA)", min_value=0.0)
        montant_investisseur_restant = st.number_input("Montant investisseur restant (FCFA)", min_value=0.0)
        code_politique = st.number_input("Code politique", min_value=0.0, max_value=1.0)

        # Bouton pour effectuer la prédiction
        if st.button("Prédire le pret et la duree maximale"):
            # Créer un DataFrame avec les données d'entrée
            input_data = pd.DataFrame({
                'montant_total_paye': [montant_total_paye],
                'montant_principal_rembourse': [montant_principal_rembourse],
                'montant_interets_rembourses': [montant_interets_rembourses],
                'montant_dernier_paiement': [montant_dernier_paiement],
                'montant_recouvrement': [montant_recouvrement],
                'montant_principal_restant': [montant_principal_restant],
                'montant_investisseur_restant': [montant_investisseur_restant],
                'code_politique': [code_politique]
            })

            # Standardiser les données
            input_data_scaled = scaler_credit_xgboost.transform(input_data)

            # Prédire le montant du prêt et la durée
            predicted_montant = xgb_model_credit.predict(input_data_scaled)
            predicted_duree = xgb_model_term.predict(input_data_scaled)

            # Afficher les résultats
            st.markdown(f'<p class="result">Montant du prêt prédit : {predicted_montant[0]:.2f} FCFA</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result">Durée du prêt prédit : {predicted_duree[0]:.2f} mois</p>', unsafe_allow_html=True)

# Afficher les informations supplémentaires
st.sidebar.header("Informations Supplémentaires")
st.sidebar.markdown("Cette application prédit la solvabilité des emprunteurs ainsi que le montant et la durée du prêt.")
st.sidebar.markdown("## Instructions")
st.sidebar.markdown("1. Remplissez tous les champs nécessaires pour la prédiction de la solvabilité.")
st.sidebar.markdown("2. Utilisez les sections pour prédire le montant et la durée du prêt.")
st.sidebar.markdown("3. Cliquez sur le bouton de prédiction pour afficher les résultats.")

# Informations de contact
st.sidebar.markdown("## Contact")
st.sidebar.markdown("Pour toute question ou support, veuillez contacter notre équipe.")
st.sidebar.markdown("Email : support@example.com")
