from types import MethodType
from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory

from .services.gnn_predictor import GNNDDIPredictor, GNNPrediction
from .views import PolypharmacyDigitalTwinView


class GNNPolypharmacyContractTests(SimpleTestCase):
	"""Regression tests for polypharmacy response contract and aggregation."""

	def _make_prediction(self, drug1, drug2, probability, interaction_type, severity):
		return GNNPrediction(
			drug1=drug1,
			drug2=drug2,
			interaction_probability=probability,
			interaction_type=interaction_type,
			confidence=0.9,
			severity=severity,
			model_used='test',
			smiles1=None,
			smiles2=None,
			fingerprint_similarity=None,
			mechanism_hypothesis=f'{drug1} + {drug2} mock mechanism'
		)

	def test_predict_polypharmacy_returns_dashboard_contract(self):
		predictor = GNNDDIPredictor.__new__(GNNDDIPredictor)

		mock_predictions = {
			frozenset(['Aspirin', 'Warfarin']): self._make_prediction('Aspirin', 'Warfarin', 0.91, 'mechanism', 'severe'),
			frozenset(['Aspirin', 'Simvastatin']): self._make_prediction('Aspirin', 'Simvastatin', 0.22, 'no_interaction', 'none'),
			frozenset(['Warfarin', 'Simvastatin']): self._make_prediction('Warfarin', 'Simvastatin', 0.64, 'effect', 'moderate'),
		}

		def fake_predict(self, drug1, drug2, smiles1=None, smiles2=None):
			return mock_predictions[frozenset([drug1, drug2])]

		predictor.predict = MethodType(fake_predict, predictor)

		result = predictor.predict_polypharmacy([
			{'name': 'Aspirin', 'smiles': ''},
			{'name': 'Warfarin', 'smiles': ''},
			{'name': 'Simvastatin', 'smiles': ''},
		])

		self.assertIn('drugs', result)
		self.assertIn('interactions', result)
		self.assertIn('total_interactions', result)
		self.assertIn('max_risk_score', result)
		self.assertIn('overall_risk_level', result)
		self.assertIn('hub_drug', result)
		self.assertIn('hub_interaction_count', result)

		self.assertEqual(result['total_interactions'], 2)
		self.assertEqual(result['hub_drug'], 'Warfarin')
		self.assertEqual(result['hub_interaction_count'], 2)
		self.assertAlmostEqual(result['max_risk_score'], 0.91)
		self.assertEqual(result['overall_risk_level'], 'critical')

		top_edge = result['interactions'][0]
		self.assertEqual(top_edge['source'], 'Aspirin')
		self.assertEqual(top_edge['target'], 'Warfarin')
		self.assertEqual(top_edge['severity'], 'severe')
		self.assertIn('affected_systems', top_edge)

	def test_predict_polypharmacy_handles_no_meaningful_edges(self):
		predictor = GNNDDIPredictor.__new__(GNNDDIPredictor)

		low_pred = self._make_prediction('Aspirin', 'Warfarin', 0.12, 'no_interaction', 'none')

		def fake_predict(self, drug1, drug2, smiles1=None, smiles2=None):
			return low_pred

		predictor.predict = MethodType(fake_predict, predictor)

		result = predictor.predict_polypharmacy([
			{'name': 'Aspirin', 'smiles': ''},
			{'name': 'Warfarin', 'smiles': ''},
			{'name': 'Simvastatin', 'smiles': ''},
		])

		self.assertEqual(result['total_interactions'], 0)
		self.assertEqual(result['interactions'], [])
		self.assertEqual(result['max_risk_score'], 0.0)
		self.assertEqual(result['overall_risk_level'], 'low')
		self.assertIsNone(result['hub_drug'])
		self.assertEqual(result['hub_interaction_count'], 0)


class PolypharmacyDigitalTwinEndpointTests(SimpleTestCase):
	"""Contract tests for the digital twin endpoint wiring."""

	def setUp(self):
		self.factory = APIRequestFactory()

	@patch('ddi_api.views.lookup_drug')
	@patch('ddi_api.views.get_polypharmacy_digital_twin_service')
	@patch('ddi_api.views.PredictionLog.objects.create')
	def test_endpoint_returns_twin_payload(self, mock_log_create, mock_get_service, mock_lookup_drug):
		mock_lookup_drug.side_effect = lambda d: {
			'name': d.get('name', 'Unknown'),
			'smiles': d.get('smiles', ''),
			'drugbank_id': d.get('drugbank_id', ''),
			'therapeutic_class': d.get('therapeutic_class', ''),
		}

		mock_service = Mock()
		mock_service.analyze.return_value = {
			'drugs': ['Aspirin', 'Warfarin', 'Simvastatin'],
			'summary': {
				'toxicity_score': 0.81,
				'risk_level': 'critical',
				'confidence_tier': 'graph-supported',
			},
			'factors': {
				'pairwise_baseline': {'score': 0.78},
				'enzyme_competition': {'score': 0.80, 'conflict_count': 2, 'conflicts': []},
				'target_overlap': {'score': 0.40, 'available': True, 'pair_count_with_overlap': 1, 'conflicts': []},
				'organ_burden': {'score': 0.60, 'systems': {'Metabolic/CYP450': 0.81}, 'top_systems': ['Metabolic/CYP450']},
				'network_stress': {'score': 0.65, 'hub_drug': 'Warfarin', 'hub_degree': 2, 'edge_density': 0.66, 'high_risk_density': 0.33},
			},
			'tension_network': {
				'nodes': [
					{'id': 'Aspirin', 'type': 'drug'},
					{'id': 'Warfarin', 'type': 'drug'},
					{'id': 'Simvastatin', 'type': 'drug'},
				],
				'edges': [],
				'hub_drug': 'Warfarin',
			},
			'body_map': {'Metabolic/CYP450': 0.81},
			'recommendations': ['Critical cumulative toxicity signal detected.'],
		}
		mock_get_service.return_value = mock_service

		request = self.factory.post(
			'/api/v1/polypharmacy-digital-twin/',
			{
				'drugs': [
					{'name': 'Aspirin'},
					{'name': 'Warfarin'},
					{'name': 'Simvastatin'},
				]
			},
			format='json'
		)

		response = PolypharmacyDigitalTwinView.as_view()(request)

		self.assertEqual(response.status_code, 200)
		self.assertIn('summary', response.data)
		self.assertIn('factors', response.data)
		self.assertIn('tension_network', response.data)
		self.assertIn('body_map', response.data)
		self.assertIn('inference_time_ms', response.data)
		self.assertEqual(response.data['summary']['risk_level'], 'critical')

		mock_service.analyze.assert_called_once()
		mock_log_create.assert_called_once()

	def test_endpoint_validates_minimum_drug_count(self):
		request = self.factory.post(
			'/api/v1/polypharmacy-digital-twin/',
			{'drugs': [{'name': 'Aspirin'}]},
			format='json'
		)

		response = PolypharmacyDigitalTwinView.as_view()(request)

		self.assertEqual(response.status_code, 400)
		self.assertIn('drugs', response.data)
