"""
Unit tests for enhanced DDI services

Tests for:
- CYP450 Database
- OpenFDA Service
- GNN Predictor
- Polypharmacy Scorer
- Ensemble Predictor
- Enhanced Data Ingestion
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add web directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "web"))


class TestCYP450Database:
    """Tests for CYP450 enzyme database."""
    
    @pytest.fixture
    def cyp450_db(self):
        """Create CYP450 database instance."""
        from ddi_api.services.cyp450_database import CYP450Database
        return CYP450Database()
    
    def test_get_drug_profile_known_drug(self, cyp450_db):
        """Test getting CYP profile for a known drug."""
        profile = cyp450_db.get_drug_cyp_profile("warfarin")
        
        assert profile is not None
        assert "CYP2C9" in profile
        assert "substrate" in profile["CYP2C9"]
    
    def test_get_drug_profile_unknown_drug(self, cyp450_db):
        """Test getting CYP profile for unknown drug returns empty."""
        profile = cyp450_db.get_drug_cyp_profile("fakemedicine123")
        
        assert profile == {}
    
    def test_get_drug_profile_case_insensitive(self, cyp450_db):
        """Test drug lookup is case insensitive."""
        profile_lower = cyp450_db.get_drug_cyp_profile("warfarin")
        profile_upper = cyp450_db.get_drug_cyp_profile("WARFARIN")
        profile_mixed = cyp450_db.get_drug_cyp_profile("Warfarin")
        
        assert profile_lower == profile_upper == profile_mixed
    
    def test_get_substrates(self, cyp450_db):
        """Test getting substrates for an enzyme."""
        substrates = cyp450_db.get_substrates("CYP3A4")
        
        assert len(substrates) > 0
        assert "simvastatin" in substrates
        assert "midazolam" in substrates
    
    def test_get_inhibitors(self, cyp450_db):
        """Test getting inhibitors for an enzyme."""
        inhibitors = cyp450_db.get_inhibitors("CYP3A4", "strong")
        
        assert len(inhibitors) > 0
        assert "ketoconazole" in inhibitors
    
    def test_get_inducers(self, cyp450_db):
        """Test getting inducers for an enzyme."""
        inducers = cyp450_db.get_inducers("CYP3A4")
        
        assert len(inducers) > 0
        assert "rifampin" in inducers
    
    def test_check_interaction_inhibitor_substrate(self, cyp450_db):
        """Test detecting inhibitor-substrate interaction."""
        # Ketoconazole (strong CYP3A4 inhibitor) + Simvastatin (CYP3A4 substrate)
        interactions = cyp450_db.check_cyp_interaction("ketoconazole", "simvastatin")
        
        assert len(interactions) > 0
        assert any(i.severity == "severe" for i in interactions)
        assert any("CYP3A4" in str(i.enzyme) for i in interactions)
    
    def test_check_interaction_inducer_substrate(self, cyp450_db):
        """Test detecting inducer-substrate interaction."""
        # Rifampin (CYP3A4 inducer) + Simvastatin (CYP3A4 substrate)
        interactions = cyp450_db.check_cyp_interaction("rifampin", "simvastatin")
        
        assert len(interactions) > 0
        assert any(i.severity in ["major", "severe"] for i in interactions)
    
    def test_check_interaction_no_interaction(self, cyp450_db):
        """Test no interaction for unrelated drugs."""
        # Metformin (not CYP metabolized) + Gabapentin (not metabolized)
        interactions = cyp450_db.check_cyp_interaction("metformin", "gabapentin")
        
        assert len(interactions) == 0
    
    def test_high_risk_combinations(self, cyp450_db):
        """Test getting high-risk drug combinations."""
        high_risk = cyp450_db.get_high_risk_combinations()
        
        assert len(high_risk) > 0
        assert all('drug1' in item and 'drug2' in item for item in high_risk)


class TestPolypharmacyScorer:
    """Tests for polypharmacy risk scoring."""
    
    @pytest.fixture
    def scorer(self):
        """Create polypharmacy scorer instance."""
        from ddi_api.services.polypharmacy_scorer import PolypharmacyRiskScorer
        return PolypharmacyRiskScorer()
    
    def test_classify_polypharmacy_minor(self, scorer):
        """Test classification with few medications."""
        level = scorer.classify_polypharmacy(3)
        assert level == "minor"
    
    def test_classify_polypharmacy_standard(self, scorer):
        """Test classification at polypharmacy threshold."""
        level = scorer.classify_polypharmacy(5)
        assert level == "polypharmacy"
    
    def test_classify_polypharmacy_excessive(self, scorer):
        """Test classification with many medications."""
        level = scorer.classify_polypharmacy(12)
        assert level == "excessive_polypharmacy"
    
    def test_identify_high_risk_medications(self, scorer):
        """Test identifying high-risk medications."""
        medications = ["warfarin", "aspirin", "metformin", "insulin"]
        high_risk, nti = scorer.identify_high_risk_medications(medications)
        
        assert "warfarin" in high_risk
        assert "warfarin" in nti  # NTI drug
        assert "insulin" in high_risk
    
    def test_detect_duplicate_therapies(self, scorer):
        """Test detecting duplicate therapies."""
        medications = ["omeprazole", "pantoprazole", "aspirin"]  # Two PPIs
        duplicates = scorer.detect_duplicate_therapies(medications)
        
        assert len(duplicates) > 0
        assert any(d['therapeutic_class'] == 'ppis' for d in duplicates)
    
    def test_no_duplicate_therapies(self, scorer):
        """Test no false positives for different classes."""
        medications = ["aspirin", "metoprolol", "atorvastatin"]
        duplicates = scorer.detect_duplicate_therapies(medications)
        
        assert len(duplicates) == 0
    
    def test_risk_assessment_empty_list(self, scorer):
        """Test risk assessment with no medications."""
        report = scorer.assess_polypharmacy_risk([])
        
        assert report.total_medications == 0
        assert report.overall_risk_score == 0.0
    
    def test_risk_assessment_low_risk(self, scorer):
        """Test risk assessment for low-risk combination."""
        medications = ["acetaminophen", "vitamin D"]
        report = scorer.assess_polypharmacy_risk(medications)
        
        assert report.total_medications == 2
        assert report.polypharmacy_level == "minor"
    
    def test_risk_assessment_high_risk(self, scorer):
        """Test risk assessment for high-risk combination."""
        # Many medications including high-risk ones
        medications = [
            "warfarin", "aspirin", "simvastatin", "ketoconazole",
            "metformin", "insulin", "lisinopril", "metoprolol",
            "omeprazole", "pantoprazole"
        ]
        report = scorer.assess_polypharmacy_risk(medications)
        
        assert report.total_medications == 10
        assert report.polypharmacy_level == "excessive_polypharmacy"
        assert report.high_risk_drug_count > 0
        assert len(report.recommendations) > 0


class TestEnhancedDataIngestion:
    """Tests for enhanced data ingestion service."""
    
    @pytest.fixture
    def ingestion(self):
        """Create enhanced data ingestion instance."""
        from ddi_api.services.enhanced_data_ingestion import EnhancedDataIngestion
        return EnhancedDataIngestion()
    
    def test_get_food_drug_interactions_all(self, ingestion):
        """Test getting all food-drug interactions."""
        interactions = ingestion.get_food_drug_interactions()
        
        assert len(interactions) > 0
        assert all(hasattr(i, 'food') and hasattr(i, 'drug') for i in interactions)
    
    def test_get_food_drug_interactions_by_drug(self, ingestion):
        """Test filtering food interactions by drug."""
        interactions = ingestion.get_food_drug_interactions(drug_name="warfarin")
        
        assert len(interactions) > 0
        assert all("warfarin" in i.drug.lower() for i in interactions)
    
    def test_get_food_drug_interactions_by_food(self, ingestion):
        """Test filtering food interactions by food."""
        interactions = ingestion.get_food_drug_interactions(food_name="grapefruit")
        
        assert len(interactions) > 0
        assert all("grapefruit" in i.food.lower() for i in interactions)
    
    def test_get_herbal_drug_interactions_all(self, ingestion):
        """Test getting all herbal-drug interactions."""
        interactions = ingestion.get_herbal_drug_interactions()
        
        assert len(interactions) > 0
    
    def test_get_herbal_drug_interactions_by_herb(self, ingestion):
        """Test filtering herbal interactions by herb."""
        interactions = ingestion.get_herbal_drug_interactions(herb_name="St. John's Wort")
        
        assert len(interactions) > 0
        assert all("john" in i.herb.lower() for i in interactions)
    
    def test_get_interaction_statistics(self, ingestion):
        """Test getting interaction statistics."""
        stats = ingestion.get_interaction_statistics()
        
        assert 'food_drug_interactions' in stats
        assert 'herbal_drug_interactions' in stats
        assert stats['food_drug_interactions'] > 0
        assert stats['herbal_drug_interactions'] > 0


class TestGNNPredictor:
    """Tests for GNN-based DDI predictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create GNN predictor instance."""
        from ddi_api.services.gnn_predictor import GNNDDIPredictor, ModelType
        return GNNDDIPredictor(ModelType.SIMPLE_MLP)
    
    @pytest.fixture
    def feature_extractor(self):
        """Create molecular feature extractor."""
        from ddi_api.services.gnn_predictor import MolecularFeatureExtractor
        return MolecularFeatureExtractor()
    
    def test_fingerprint_from_valid_smiles(self, feature_extractor):
        """Test fingerprint generation from valid SMILES."""
        # Aspirin SMILES
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        fp = feature_extractor.smiles_to_fingerprint(smiles)
        
        assert fp is not None
        assert len(fp) == 2048  # Default fingerprint size
        assert all(b in [0, 1] for b in fp)  # Binary fingerprint
    
    def test_fingerprint_from_empty_smiles(self, feature_extractor):
        """Test fingerprint generation from empty SMILES."""
        fp = feature_extractor.smiles_to_fingerprint("")
        assert fp is None
    
    def test_tanimoto_similarity_identical(self, feature_extractor):
        """Test Tanimoto similarity for identical molecules."""
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        similarity = feature_extractor.calculate_tanimoto_similarity(smiles, smiles)
        
        assert similarity is not None
        assert similarity == 1.0  # Identical molecules
    
    def test_tanimoto_similarity_different(self, feature_extractor):
        """Test Tanimoto similarity for different molecules."""
        aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        
        similarity = feature_extractor.calculate_tanimoto_similarity(aspirin, ibuprofen)
        
        assert similarity is not None
        assert 0 < similarity < 1  # Similar but not identical
    
    def test_predict_with_smiles(self, predictor):
        """Test prediction with SMILES input."""
        aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        warfarin = "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"
        
        prediction = predictor.predict("aspirin", "warfarin", aspirin, warfarin)
        
        assert prediction is not None
        assert prediction.drug1 == "aspirin"
        assert prediction.drug2 == "warfarin"
        assert prediction.interaction_type in predictor.INTERACTION_TYPES
        assert 0 <= prediction.confidence <= 1
    
    def test_predict_without_smiles(self, predictor):
        """Test prediction without SMILES (heuristic fallback)."""
        prediction = predictor.predict("aspirin", "warfarin")
        
        assert prediction is not None
        assert prediction.model_used == "heuristic"
        assert prediction.interaction_type in predictor.INTERACTION_TYPES


class TestEnsemblePredictor:
    """Tests for ensemble DDI predictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create ensemble predictor instance."""
        from ddi_api.services.ensemble_predictor import EnsembleDDIPredictor
        return EnsembleDDIPredictor()
    
    def test_get_available_sources(self, predictor):
        """Test checking available prediction sources."""
        sources = predictor.get_available_sources()
        
        assert 'cyp450' in sources
        assert 'gnn' in sources
        assert isinstance(sources['cyp450'], bool)
    
    def test_predict_returns_ensemble_prediction(self, predictor):
        """Test predict returns EnsemblePrediction object."""
        prediction = predictor.predict("aspirin", "warfarin")
        
        assert prediction is not None
        assert prediction.drug1 == "aspirin"
        assert prediction.drug2 == "warfarin"
        assert hasattr(prediction, 'final_severity')
        assert hasattr(prediction, 'source_predictions')
        assert hasattr(prediction, 'recommendations')
    
    def test_predict_includes_cyp450(self, predictor):
        """Test prediction includes CYP450 source when available."""
        prediction = predictor.predict("ketoconazole", "simvastatin")
        
        cyp450_pred = next(
            (p for p in prediction.source_predictions if p.source.value == 'cyp450'),
            None
        )
        
        assert cyp450_pred is not None
        # This is a known severe CYP3A4 interaction
        if cyp450_pred.available:
            assert cyp450_pred.severity in ['major', 'severe']
    
    def test_predict_has_recommendations(self, predictor):
        """Test prediction includes recommendations."""
        prediction = predictor.predict("ketoconazole", "simvastatin")
        
        assert len(prediction.recommendations) > 0


class TestPatientProfile:
    """Tests for patient profile risk adjustments."""
    
    @pytest.fixture
    def patient_profile_class(self):
        """Get PatientProfile class."""
        from ddi_api.services.polypharmacy_scorer import PatientProfile
        return PatientProfile
    
    def test_age_risk_factor_elderly(self, patient_profile_class):
        """Test age risk factor for elderly patient."""
        patient = patient_profile_class(age=75)
        factor = patient.get_age_risk_factor()
        
        assert factor > 1.0
    
    def test_age_risk_factor_adult(self, patient_profile_class):
        """Test age risk factor for adult patient."""
        patient = patient_profile_class(age=35)
        factor = patient.get_age_risk_factor()
        
        assert factor == 1.0
    
    def test_renal_risk_factor_impaired(self, patient_profile_class):
        """Test renal risk factor for impaired kidney function."""
        patient = patient_profile_class(creatinine_clearance=25)
        factor = patient.get_renal_risk_factor()
        
        assert factor > 1.0
    
    def test_hepatic_risk_factor(self, patient_profile_class):
        """Test hepatic risk factor."""
        patient = patient_profile_class(hepatic_function="moderate")
        factor = patient.get_hepatic_risk_factor()
        
        assert factor > 1.0


class TestOfflineTrainingData:
    """Tests for offline training data module."""
    
    @pytest.fixture
    def offline_data(self):
        """Import offline training data module."""
        from ddi_api.services.offline_training_data import (
            get_all_drugs, get_all_interactions, get_drug_by_id,
            get_drug_by_name, get_interaction, get_training_statistics
        )
        return {
            'get_all_drugs': get_all_drugs,
            'get_all_interactions': get_all_interactions,
            'get_drug_by_id': get_drug_by_id,
            'get_drug_by_name': get_drug_by_name,
            'get_interaction': get_interaction,
            'get_training_statistics': get_training_statistics
        }
    
    def test_get_all_drugs(self, offline_data):
        """Test getting all drugs from offline database."""
        drugs = offline_data['get_all_drugs']()
        
        assert len(drugs) > 30  # We have 40+ drugs
        assert all(hasattr(d, 'drugbank_id') for d in drugs)
        assert all(hasattr(d, 'smiles') for d in drugs)
    
    def test_get_all_interactions(self, offline_data):
        """Test getting all interactions."""
        interactions = offline_data['get_all_interactions']()
        
        assert len(interactions) > 30  # We have 40+ interactions
        assert all(hasattr(i, 'drug1_id') for i in interactions)
        assert all(hasattr(i, 'severity') for i in interactions)
    
    def test_get_drug_by_id(self, offline_data):
        """Test getting drug by DrugBank ID."""
        drug = offline_data['get_drug_by_id']("DB00682")  # Warfarin
        
        assert drug is not None
        assert drug.name == "Warfarin"
        assert drug.smiles is not None
    
    def test_get_drug_by_name(self, offline_data):
        """Test getting drug by name."""
        drug = offline_data['get_drug_by_name']("Warfarin")
        
        assert drug is not None
        assert drug.drugbank_id == "DB00682"
    
    def test_get_drug_by_name_case_insensitive(self, offline_data):
        """Test drug name lookup is case-insensitive."""
        drug1 = offline_data['get_drug_by_name']("warfarin")
        drug2 = offline_data['get_drug_by_name']("WARFARIN")
        
        assert drug1 is not None
        assert drug2 is not None
        assert drug1.drugbank_id == drug2.drugbank_id
    
    def test_get_interaction(self, offline_data):
        """Test getting interaction between two drugs."""
        interaction = offline_data['get_interaction']("DB00682", "DB00945")  # Warfarin-Aspirin
        
        assert interaction is not None
        assert interaction.severity == "severe"
    
    def test_get_interaction_reversed(self, offline_data):
        """Test interaction lookup works in both directions."""
        int1 = offline_data['get_interaction']("DB00682", "DB00945")
        int2 = offline_data['get_interaction']("DB00945", "DB00682")
        
        assert int1 is not None
        assert int2 is not None
        assert int1.severity == int2.severity
    
    def test_get_training_statistics(self, offline_data):
        """Test getting training data statistics."""
        stats = offline_data['get_training_statistics']()
        
        assert 'total_drugs' in stats
        assert 'total_interactions' in stats
        assert 'severity_distribution' in stats
        assert stats['total_drugs'] > 30
        assert stats['total_interactions'] > 30
    
    def test_drugs_have_smiles(self, offline_data):
        """Test that all drugs have SMILES structures."""
        drugs = offline_data['get_all_drugs']()
        drugs_with_smiles = [d for d in drugs if d.smiles]
        
        # All drugs should have SMILES for training
        assert len(drugs_with_smiles) == len(drugs)
    
    def test_severity_distribution(self, offline_data):
        """Test severity distribution in interactions."""
        stats = offline_data['get_training_statistics']()
        severity_dist = stats['severity_distribution']
        
        assert 'severe' in severity_dist
        assert 'major' in severity_dist
        assert severity_dist['severe'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
