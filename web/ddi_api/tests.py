import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType
from unittest.mock import Mock, patch

from django.core.management import call_command
from django.test import SimpleTestCase
from rest_framework.test import APIRequestFactory

from .services.calibration_metrics import expected_calibration_error, generate_calibration_report
from .services.gnn_predictor import GNNDDIPredictor, GNNPrediction
from .views import DDIPredictionView, DatabaseStatsView, CalibrationMetricsView, PolypharmacyDigitalTwinView


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


class DDIPredictionObservabilityContractTests(SimpleTestCase):
	"""Regression tests for provenance and score observability in /predict/."""

	def setUp(self):
		self.factory = APIRequestFactory()

	@patch('ddi_api.views.lookup_drug')
	@patch('ddi_api.views.get_gnn_predictor')
	@patch('ddi_api.views.PredictionLog.objects.create')
	def test_predict_ai_path_includes_raw_calibrated_and_provenance(self, mock_log_create, mock_get_predictor, mock_lookup_drug):
		mock_lookup_drug.side_effect = lambda d: {
			'name': d.get('name', 'Unknown'),
			'smiles': d.get('smiles', ''),
			'drugbank_id': '',
			'therapeutic_class': d.get('therapeutic_class', ''),
		}

		mock_predictor = Mock()
		mock_predictor.predict.return_value = GNNPrediction(
			drug1='AlphaDrug',
			drug2='BetaDrug',
			interaction_probability=0.81,
			interaction_type='effect',
			confidence=0.92,
			severity='major',
			model_used='trained_gnn',
			smiles1='CCO',
			smiles2='CCN',
			fingerprint_similarity=0.44,
			mechanism_hypothesis='Mock mechanism',
			raw_interaction_probability=0.77,
			calibration_method='platt_scaling',
			calibration_version='platt_a=1.0;platt_b=0.0',
			fallback_reason=None,
			provenance={'prediction_path': 'trained_gnn'}
		)
		mock_get_predictor.return_value = mock_predictor

		request = self.factory.post(
			'/api/v1/predict/',
			{
				'drug_a': {'name': 'AlphaDrug'},
				'drug_b': {'name': 'BetaDrug'},
				'include_explanation': True,
			},
			format='json'
		)

		response = DDIPredictionView.as_view()(request)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.data['risk_score'], 0.81)
		self.assertEqual(response.data['raw_score'], 0.77)
		self.assertEqual(response.data['calibrated_score'], 0.81)
		self.assertIn('provenance', response.data)
		self.assertEqual(response.data['provenance']['calibration_method'], 'platt_scaling')
		self.assertIn('explanation', response.data)
		self.assertEqual(response.data['explanation']['calibration']['method'], 'platt_scaling')

		mock_log_create.assert_called_once()
		logged = mock_log_create.call_args.kwargs
		self.assertEqual(logged['risk_score'], 0.81)
		self.assertEqual(logged['raw_score'], 0.77)
		self.assertEqual(logged['calibrated_score'], 0.81)
		self.assertIn('provenance', logged)

	@patch('ddi_api.views.lookup_drug')
	@patch('ddi_api.views.PredictionLog.objects.create')
	def test_predict_known_interaction_has_rule_provenance(self, mock_log_create, mock_lookup_drug):
		def _lookup(drug_input):
			name = drug_input.get('name', '')
			if name.lower() == 'aspirin':
				return {'name': 'Aspirin', 'smiles': '', 'drugbank_id': '', 'therapeutic_class': 'NSAID'}
			return {'name': 'Warfarin', 'smiles': '', 'drugbank_id': '', 'therapeutic_class': 'Anticoagulant'}

		mock_lookup_drug.side_effect = _lookup

		request = self.factory.post(
			'/api/v1/predict/',
			{
				'drug_a': {'name': 'Aspirin'},
				'drug_b': {'name': 'Warfarin'},
				'include_explanation': True,
			},
			format='json'
		)

		response = DDIPredictionView.as_view()(request)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.data['source'], 'categorical_rule_engine')
		self.assertIn('provenance', response.data)
		self.assertEqual(response.data['provenance']['prediction_path'], 'categorical_rule_engine')
		self.assertEqual(response.data['provenance']['calibration_method'], 'none')
		self.assertEqual(response.data['raw_score'], response.data['calibrated_score'])
		mock_log_create.assert_called_once()


class DatabaseStatsRecentPredictionsRegressionTests(SimpleTestCase):
	"""Regression test to ensure dashboard stats query uses created_at field."""

	def setUp(self):
		self.factory = APIRequestFactory()

	@patch('ddi_api.views.KnowledgeGraphService.is_connected', return_value=False)
	@patch('ddi_api.views.PredictionLog.objects.filter')
	def test_recent_predictions_uses_created_at_filter(self, mock_filter, _mock_kg_connected):
		mock_filter.return_value.count.return_value = 7

		request = self.factory.get('/api/v1/stats/')
		response = DatabaseStatsView.as_view()(request)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.data['recent_predictions'], 7)
		self.assertIn('created_at__gte', mock_filter.call_args.kwargs)


class CalibrationMetricsServiceTests(SimpleTestCase):
	"""Unit tests for calibration metric utilities."""

	def test_expected_calibration_error_perfect_alignment_is_zero(self):
		labels = [0, 0, 0, 1, 1, 1]
		scores = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

		ece, mce, bins = expected_calibration_error(labels, scores, n_bins=5)

		self.assertEqual(ece, 0.0)
		self.assertEqual(mce, 0.0)
		self.assertEqual(len(bins), 5)

	def test_generate_report_shows_improvement_for_better_calibrated_scores(self):
		labels = [0, 0, 0, 1, 1, 1]
		raw_scores = [0.40, 0.45, 0.35, 0.55, 0.60, 0.65]
		calibrated_scores = [0.08, 0.10, 0.20, 0.80, 0.90, 0.92]

		report = generate_calibration_report(
			labels=labels,
			raw_scores=raw_scores,
			calibrated_scores=calibrated_scores,
			n_bins=5,
			n_bootstrap=200,
			seed=7,
		)

		self.assertIn('raw', report)
		self.assertIn('calibrated', report)
		self.assertIn('delta', report)
		self.assertGreater(report['delta']['ece_improvement'], 0)
		self.assertGreater(report['delta']['brier_improvement'], 0)


class CalibrationReportCommandTests(SimpleTestCase):
	"""Integration test for calibration report management command."""

	def test_generate_calibration_report_command_outputs_json(self):
		csv_content = "\n".join([
			"label,raw_score,calibrated_score",
			"0,0.40,0.05",
			"0,0.35,0.08",
			"1,0.60,0.90",
			"1,0.55,0.88",
			"1,0.70,0.95",
			"0,0.30,0.12",
		])

		with TemporaryDirectory() as temp_dir:
			temp_path = Path(temp_dir)
			csv_path = temp_path / 'calibration_eval.csv'
			out_path = temp_path / 'report.json'
			csv_path.write_text(csv_content, encoding='utf-8')

			call_command(
				'generate_calibration_report',
				'--input-csv', str(csv_path),
				'--output-json', str(out_path),
				'--bins', '5',
				'--bootstrap', '200',
				'--seed', '11',
			)

			self.assertTrue(out_path.exists())

			report = json.loads(out_path.read_text(encoding='utf-8'))
			self.assertIn('meta', report)
			self.assertIn('raw', report)
			self.assertIn('calibrated', report)
			self.assertIn('delta', report)
			self.assertEqual(report['meta']['n_samples'], 6)
			self.assertIn('ece_confidence_interval', report['raw'])
			self.assertIn('brier_confidence_interval', report['calibrated'])


class CalibrationMetricsEndpointTests(SimpleTestCase):
	"""Contract tests for POST /api/v1/calibration/metrics/."""

	def setUp(self):
		self.factory = APIRequestFactory()

	def test_endpoint_returns_calibration_report(self):
		request = self.factory.post(
			'/api/v1/calibration/metrics/',
			{
				'labels': [0, 0, 1, 1, 1, 0],
				'raw_scores': [0.40, 0.35, 0.60, 0.55, 0.70, 0.30],
				'calibrated_scores': [0.10, 0.08, 0.90, 0.88, 0.95, 0.12],
				'n_bins': 5,
				'n_bootstrap': 200,
				'seed': 13,
			},
			format='json',
		)

		response = CalibrationMetricsView.as_view()(request)

		self.assertEqual(response.status_code, 200)
		self.assertIn('meta', response.data)
		self.assertIn('raw', response.data)
		self.assertIn('calibrated', response.data)
		self.assertIn('delta', response.data)
		self.assertGreater(response.data['delta']['ece_improvement'], 0)

	def test_endpoint_rejects_non_array_payload(self):
		request = self.factory.post(
			'/api/v1/calibration/metrics/',
			{
				'labels': 'not-an-array',
				'raw_scores': [0.2, 0.8],
				'calibrated_scores': [0.1, 0.9],
			},
			format='json',
		)

		response = CalibrationMetricsView.as_view()(request)

		self.assertEqual(response.status_code, 400)
		self.assertIn('error', response.data)
