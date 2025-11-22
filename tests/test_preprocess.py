# tests/test_preprocess.py

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ajouter le dossier scripts au path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from preprocess import build_features


@pytest.fixture
def sample_dataframe():
    """
    Crée un DataFrame de test avec les colonnes nécessaires.
    """
    np.random.seed(42)
    data = {
        "Engine rpm": [1500, 2000, 2500, 3000, 3500],
        "Lub oil pressure": [40, 45, 50, 55, 60],
        "Coolant temp": [90, 95, 100, 105, 110],
        "lub oil temp": [80, 85, 90, 95, 100],
        "Engine Condition": [0, 0, 1, 1, 1]
    }
    return pd.DataFrame(data)


@pytest.fixture
def incomplete_dataframe():
    """
    DataFrame avec colonnes manquantes.
    """
    data = {
        "Engine rpm": [1500, 2000],
        "Other_col": [1, 2]
    }
    return pd.DataFrame(data)


class TestBuildFeatures:
    """
    Tests pour la fonction build_features.
    """
    
    def test_creates_engine_power(self, sample_dataframe):
        """
        Teste que la feature Engine_power est correctement créée.
        """
        df_processed = build_features(sample_dataframe)
        
        assert "Engine_power" in df_processed.columns
        assert df_processed["Engine_power"].iloc[0] == 1500 * 40
        assert df_processed["Engine_power"].iloc[1] == 2000 * 45
    
    def test_creates_temperature_difference(self, sample_dataframe):
        """
        Teste que la feature Temperature_difference est correctement créée.
        """
        df_processed = build_features(sample_dataframe)
        
        assert "Temperature_difference" in df_processed.columns
        assert df_processed["Temperature_difference"].iloc[0] == 90 - 80
        assert df_processed["Temperature_difference"].iloc[1] == 95 - 85
    
    def test_handles_missing_columns(self, incomplete_dataframe):
        """
        Teste le comportement avec des colonnes manquantes.
        """
        # Ne devrait pas crasher
        df_processed = build_features(incomplete_dataframe)
        
        # Les features ne devraient pas être créées
        assert "Engine_power" not in df_processed.columns
        assert "Temperature_difference" not in df_processed.columns
    
    def test_preserves_original_columns(self, sample_dataframe):
        """
        Teste que les colonnes originales sont préservées.
        """
        original_cols = set(sample_dataframe.columns)
        df_processed = build_features(sample_dataframe)
        
        # Toutes les colonnes originales doivent être présentes
        for col in original_cols:
            assert col in df_processed.columns
    
    def test_no_nulls_created(self, sample_dataframe):
        """
        Teste qu'aucune valeur NaN n'est créée par les features.
        """
        df_processed = build_features(sample_dataframe)
        
        # Vérifier qu'il n'y a pas de NaN dans les nouvelles features
        assert not df_processed["Engine_power"].isna().any()
        assert not df_processed["Temperature_difference"].isna().any()
    
    def test_correct_dtypes(self, sample_dataframe):
        """
        Teste que les types de données sont corrects.
        """
        df_processed = build_features(sample_dataframe)
        
        # Les features numériques doivent rester numériques
        assert pd.api.types.is_numeric_dtype(df_processed["Engine_power"])
        assert pd.api.types.is_numeric_dtype(df_processed["Temperature_difference"])
    
    def test_feature_values_positive(self, sample_dataframe):
        """
        Teste que Engine_power contient uniquement des valeurs positives.
        """
        df_processed = build_features(sample_dataframe)
        
        assert (df_processed["Engine_power"] > 0).all()
    
    def test_temperature_difference_logical(self, sample_dataframe):
        """
        Teste que la différence de température est logique.
        """
        df_processed = build_features(sample_dataframe)
        
        # La différence devrait généralement être positive
        # (coolant temp > lub oil temp dans des conditions normales)
        assert df_processed["Temperature_difference"].mean() > 0
    
    def test_dataframe_shape_unchanged(self, sample_dataframe):
        """
        Teste que le nombre de lignes ne change pas.
        """
        original_rows = len(sample_dataframe)
        df_processed = build_features(sample_dataframe)
        
        assert len(df_processed) == original_rows
    
    def test_features_correlation(self, sample_dataframe):
        """
        Teste que les features ont une corrélation logique.
        """
        df_processed = build_features(sample_dataframe)
        
        # Engine_power devrait être corrélé avec Engine rpm
        correlation = df_processed[["Engine rpm", "Engine_power"]].corr().iloc[0, 1]
        assert correlation > 0.5  # Corrélation positive forte


class TestEdgeCases:
    """
    Tests pour les cas limites.
    """
    
    def test_empty_dataframe(self):
        """
        Teste le comportement avec un DataFrame vide.
        """
        df_empty = pd.DataFrame()
        df_processed = build_features(df_empty)
        
        assert len(df_processed) == 0
    
    def test_single_row(self):
        """
        Teste avec une seule ligne de données.
        """
        df_single = pd.DataFrame({
            "Engine rpm": [1500],
            "Lub oil pressure": [40],
            "Coolant temp": [90],
            "lub oil temp": [80]
        })
        
        df_processed = build_features(df_single)
        
        assert len(df_processed) == 1
        assert "Engine_power" in df_processed.columns
        assert "Temperature_difference" in df_processed.columns
    
    def test_large_values(self):
        """
        Teste avec de grandes valeurs.
        """
        df_large = pd.DataFrame({
            "Engine rpm": [10000],
            "Lub oil pressure": [1000],
            "Coolant temp": [200],
            "lub oil temp": [150]
        })
        
        df_processed = build_features(df_large)
        
        # Pas d'overflow
        assert df_processed["Engine_power"].iloc[0] == 10000 * 1000
        assert df_processed["Temperature_difference"].iloc[0] == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])