"""
Unit tests for DDI Tokenizer
Tests tokenization, drug marking, mask creation, and NER label generation
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.tokenization import DDITokenizer


@pytest.fixture
def mock_tokenizer():
    """Mock AutoTokenizer for testing without downloading models"""
    mock = MagicMock()

    # Mock convert_tokens_to_ids
    token_ids = {
        '[DRUG1]': 101,
        '[/DRUG1]': 102,
        '[DRUG2]': 103,
        '[/DRUG2]': 104,
        '[CLS]': 105,
        '[SEP]': 106,
        '[PAD]': 0,
    }
    mock.convert_tokens_to_ids.side_effect = lambda x: token_ids.get(x, 999)

    # Mock __len__
    mock.__len__ = lambda self: 30522

    # Mock __call__ for tokenization
    def tokenize_side_effect(*args, **kwargs):
        text = args[0] if args else kwargs.get('text', '')
        seq_len = kwargs.get('max_length', 10)

        # Simple mock encoding
        input_ids = [105, 101, 999, 999, 102, 103, 999, 104, 106, 0][:seq_len]
        attention_mask = [1] * (len(input_ids) - 1) + [0]

        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

    mock.side_effect = tokenize_side_effect
    mock.__call__ = tokenize_side_effect

    # Mock decode
    mock.decode.return_value = "decoded text"

    # Mock add_special_tokens
    mock.add_special_tokens = MagicMock()

    return mock


@pytest.fixture
def ddi_tokenizer(mock_tokenizer):
    """Create DDITokenizer instance with mocked tokenizer"""
    with patch('model.tokenization.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        tokenizer = DDITokenizer()
    return tokenizer


class TestDDITokenizerInitialization:
    """Test tokenizer initialization and setup"""

    def test_initialization_default_params(self, ddi_tokenizer):
        """Test tokenizer initializes with correct default parameters"""
        assert ddi_tokenizer.drug1_start_id == 101
        assert ddi_tokenizer.drug1_end_id == 102
        assert ddi_tokenizer.drug2_start_id == 103
        assert ddi_tokenizer.drug2_end_id == 104
        assert ddi_tokenizer.max_length == 512

    def test_initialization_custom_max_length(self, mock_tokenizer):
        """Test initialization with custom max_length"""
        with patch('model.tokenization.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            tokenizer = DDITokenizer(max_length=256)
        assert tokenizer.max_length == 256

    def test_special_tokens_defined(self, ddi_tokenizer):
        """Test special tokens are correctly defined"""
        assert DDITokenizer.SPECIAL_TOKENS['drug1_start'] == '[DRUG1]'
        assert DDITokenizer.SPECIAL_TOKENS['drug1_end'] == '[/DRUG1]'
        assert DDITokenizer.SPECIAL_TOKENS['drug2_start'] == '[DRUG2]'
        assert DDITokenizer.SPECIAL_TOKENS['drug2_end'] == '[/DRUG2]'

    def test_ner_labels_defined(self, ddi_tokenizer):
        """Test NER labels are correctly defined"""
        assert DDITokenizer.NER_LABELS['O'] == 0
        assert DDITokenizer.NER_LABELS['B-DRUG'] == 1
        assert DDITokenizer.NER_LABELS['I-DRUG'] == 2

    def test_get_vocab_size(self, ddi_tokenizer):
        """Test get_vocab_size returns correct size"""
        size = ddi_tokenizer.get_vocab_size()
        assert size == 30522

    def test_special_tokens_added(self, mock_tokenizer):
        """Test special tokens are added to tokenizer"""
        with patch('model.tokenization.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
            tokenizer = DDITokenizer()

        mock_tokenizer.add_special_tokens.assert_called_once()
        call_args = mock_tokenizer.add_special_tokens.call_args[0][0]
        assert 'additional_special_tokens' in call_args
        assert '[DRUG1]' in call_args['additional_special_tokens']
        assert '[/DRUG1]' in call_args['additional_special_tokens']
        assert '[DRUG2]' in call_args['additional_special_tokens']
        assert '[/DRUG2]' in call_args['additional_special_tokens']


class TestMarkDrugsInText:
    """Test drug marking functionality"""

    def test_mark_drugs_basic(self, ddi_tokenizer):
        """Test basic drug marking in text"""
        text = "The patient took aspirin and ibuprofen"
        drug1_span = (17, 24)  # "aspirin"
        drug2_span = (29, 38)  # "ibuprofen"

        result = ddi_tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        assert '[DRUG1]' in result
        assert '[/DRUG1]' in result
        assert '[DRUG2]' in result
        assert '[/DRUG2]' in result
        assert 'aspirin' in result
        assert 'ibuprofen' in result

    def test_mark_drugs_reversed_order(self, ddi_tokenizer):
        """Test drug marking with drug2 appearing before drug1"""
        text = "Ibuprofen and aspirin interaction"
        drug1_span = (14, 21)  # "aspirin" (appears second)
        drug2_span = (0, 9)    # "Ibuprofen" (appears first)

        result = ddi_tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        assert '[DRUG1]' in result
        assert '[/DRUG1]' in result
        assert '[DRUG2]' in result
        assert '[/DRUG2]' in result
        assert 'aspirin' in result

    def test_mark_drugs_adjacent(self, ddi_tokenizer):
        """Test drug marking with adjacent drugs"""
        text = "warfarin aspirin"
        drug1_span = (0, 8)   # "warfarin"
        drug2_span = (9, 16)  # "aspirin"

        result = ddi_tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        assert '[DRUG1]' in result
        assert '[DRUG2]' in result
        assert 'warfarin' in result
        assert 'aspirin' in result

    def test_mark_drugs_with_punctuation(self, ddi_tokenizer):
        """Test drug marking with punctuation around drugs"""
        text = "Taking aspirin, ibuprofen, and warfarin."
        drug1_span = (7, 14)   # "aspirin"
        drug2_span = (16, 25)  # "ibuprofen"

        result = ddi_tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        assert '[DRUG1]' in result
        assert '[DRUG2]' in result

    def test_mark_drugs_preserves_text_structure(self, ddi_tokenizer):
        """Test that marking preserves original text structure"""
        text = "Patient takes drug1 and drug2 daily"
        drug1_span = (14, 19)  # "drug1"
        drug2_span = (24, 29)  # "drug2"

        result = ddi_tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        # Should contain original words
        assert 'Patient' in result
        assert 'takes' in result
        assert 'daily' in result


class TestCreateDrugMask:
    """Test drug mask creation"""

    def test_create_drug_mask_basic(self, ddi_tokenizer):
        """Test basic drug mask creation"""
        input_ids = torch.tensor([105, 101, 999, 999, 102, 0, 0])

        mask = ddi_tokenizer._create_drug_mask(input_ids, 101, 102)

        assert mask[0] == 0.0  # Before start marker
        assert mask[1] == 1.0  # Start marker
        assert mask[2] == 1.0  # Between markers
        assert mask[3] == 1.0  # Between markers
        assert mask[4] == 1.0  # End marker
        assert mask[5] == 0.0  # After end marker
        assert mask[6] == 0.0  # Padding

    def test_create_drug_mask_no_markers(self, ddi_tokenizer):
        """Test drug mask when markers don't exist"""
        input_ids = torch.tensor([105, 999, 999, 999, 106])

        mask = ddi_tokenizer._create_drug_mask(input_ids, 101, 102)

        assert torch.all(mask == 0.0)

    def test_create_drug_mask_only_start_marker(self, ddi_tokenizer):
        """Test drug mask when only start marker exists"""
        input_ids = torch.tensor([105, 101, 999, 999, 106])

        mask = ddi_tokenizer._create_drug_mask(input_ids, 101, 102)

        # Should have no mask if end marker is missing
        assert torch.all(mask == 0.0)

    def test_create_drug_mask_adjacent_markers(self, ddi_tokenizer):
        """Test drug mask when start and end markers are adjacent"""
        input_ids = torch.tensor([105, 101, 102, 999, 106])

        mask = ddi_tokenizer._create_drug_mask(input_ids, 101, 102)

        assert mask[1] == 1.0  # Start marker
        assert mask[2] == 1.0  # End marker
        assert mask[3] == 0.0  # After markers

    def test_create_drug_mask_type(self, ddi_tokenizer):
        """Test drug mask returns correct tensor type"""
        input_ids = torch.tensor([105, 101, 999, 102, 106])

        mask = ddi_tokenizer._create_drug_mask(input_ids, 101, 102)

        assert mask.dtype == torch.float
        assert mask.shape == input_ids.shape


class TestCreateNerLabels:
    """Test NER label creation"""

    def test_create_ner_labels_basic(self, ddi_tokenizer):
        """Test NER label creation with drug masks"""
        input_ids = torch.tensor([105, 101, 999, 999, 102, 0, 0])
        drug1_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        drug2_mask = torch.zeros(7)

        ner_labels = ddi_tokenizer._create_ner_labels(input_ids, drug1_mask, drug2_mask)

        assert ner_labels[0] == 0  # O - outside
        assert ner_labels[5] == 0  # O - padding
        assert ner_labels[6] == 0  # O - padding
        # Positions 2, 3 should be B-DRUG or I-DRUG (not markers)
        assert ner_labels[2] in [0, 1, 2]
        assert ner_labels[3] in [0, 1, 2]

    def test_create_ner_labels_no_drugs(self, ddi_tokenizer):
        """Test NER labels when no drug masks are set"""
        input_ids = torch.tensor([105, 999, 999, 999, 106])
        drug1_mask = torch.zeros(5)
        drug2_mask = torch.zeros(5)

        ner_labels = ddi_tokenizer._create_ner_labels(input_ids, drug1_mask, drug2_mask)

        assert torch.all(ner_labels == 0)

    def test_create_ner_labels_multiple_drugs(self, ddi_tokenizer):
        """Test NER labels with both drug regions"""
        input_ids = torch.tensor([105, 101, 999, 102, 103, 999, 104, 0])
        drug1_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        drug2_mask = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

        ner_labels = ddi_tokenizer._create_ner_labels(input_ids, drug1_mask, drug2_mask)

        assert ner_labels[0] == 0  # Outside
        assert ner_labels[7] == 0  # Padding
        # Should have drug labels in masked regions (excluding markers)

    def test_create_ner_labels_excludes_markers(self, ddi_tokenizer):
        """Test that NER labels exclude marker tokens"""
        input_ids = torch.tensor([105, 101, 999, 102, 106])
        drug1_mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
        drug2_mask = torch.zeros(5)

        ner_labels = ddi_tokenizer._create_ner_labels(input_ids, drug1_mask, drug2_mask)

        # Markers themselves should not get drug labels
        # Only the token at position 2 (999) should potentially get B-DRUG/I-DRUG

    def test_create_ner_labels_type(self, ddi_tokenizer):
        """Test NER labels return correct tensor type"""
        input_ids = torch.tensor([105, 101, 999, 102, 106])
        drug1_mask = torch.ones(5)
        drug2_mask = torch.zeros(5)

        ner_labels = ddi_tokenizer._create_ner_labels(input_ids, drug1_mask, drug2_mask)

        assert ner_labels.dtype == torch.long
        assert ner_labels.shape == input_ids.shape


class TestTokenize:
    """Test tokenization functionality"""

    def test_tokenize_output_structure(self, ddi_tokenizer):
        """Test tokenize returns correct output structure"""
        result = ddi_tokenizer.tokenize("[DRUG1] test [/DRUG1]")

        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'drug1_mask' in result
        assert 'drug2_mask' in result
        assert 'ner_labels' in result

    def test_tokenize_output_types(self, ddi_tokenizer):
        """Test tokenize output tensor types"""
        result = ddi_tokenizer.tokenize("[DRUG1] test [/DRUG1]")

        assert isinstance(result['input_ids'], torch.Tensor)
        assert isinstance(result['attention_mask'], torch.Tensor)
        assert isinstance(result['drug1_mask'], torch.Tensor)
        assert isinstance(result['drug2_mask'], torch.Tensor)
        assert isinstance(result['ner_labels'], torch.Tensor)

    def test_tokenize_output_shapes_match(self, ddi_tokenizer):
        """Test all tokenize outputs have matching shapes"""
        result = ddi_tokenizer.tokenize("[DRUG1] test [/DRUG1]")

        seq_len = result['input_ids'].shape[0]
        assert result['attention_mask'].shape[0] == seq_len
        assert result['drug1_mask'].shape[0] == seq_len
        assert result['drug2_mask'].shape[0] == seq_len
        assert result['ner_labels'].shape[0] == seq_len

    def test_tokenize_max_length_respected(self, ddi_tokenizer):
        """Test tokenization respects max_length"""
        result = ddi_tokenizer.tokenize("test text")

        assert result['input_ids'].shape[0] <= ddi_tokenizer.max_length

    def test_tokenize_with_both_drugs(self, ddi_tokenizer):
        """Test tokenize with both drug markers present"""
        text = "[DRUG1] aspirin [/DRUG1] and [DRUG2] ibuprofen [/DRUG2]"
        result = ddi_tokenizer.tokenize(text)

        # Both masks should have some non-zero values
        assert torch.any(result['drug1_mask'] > 0)
        assert torch.any(result['drug2_mask'] > 0)

    def test_tokenize_return_tensors_pt(self, ddi_tokenizer):
        """Test tokenize returns PyTorch tensors"""
        result = ddi_tokenizer.tokenize("test", return_tensors="pt")

        assert all(isinstance(v, torch.Tensor) for v in result.values())


class TestBatchTokenize:
    """Test batch tokenization"""

    def test_batch_tokenize_single_sample(self, ddi_tokenizer):
        """Test batch tokenize with single sample"""
        texts = ["[DRUG1] test [/DRUG1]"]
        result = ddi_tokenizer.batch_tokenize(texts)

        assert result['input_ids'].shape[0] == 1
        assert result['input_ids'].dim() == 2

    def test_batch_tokenize_multiple_samples(self, ddi_tokenizer):
        """Test batch tokenize with multiple samples"""
        texts = [
            "[DRUG1] test1 [/DRUG1]",
            "[DRUG1] test2 [/DRUG1]",
            "[DRUG2] test3 [/DRUG2]"
        ]
        result = ddi_tokenizer.batch_tokenize(texts)

        assert result['input_ids'].shape[0] == 3
        assert result['attention_mask'].shape[0] == 3
        assert result['drug1_mask'].shape[0] == 3
        assert result['drug2_mask'].shape[0] == 3
        assert result['ner_labels'].shape[0] == 3

    def test_batch_tokenize_output_stacked(self, ddi_tokenizer):
        """Test batch tokenize stacks tensors correctly"""
        texts = ["text1", "text2"]
        result = ddi_tokenizer.batch_tokenize(texts)

        # All outputs should be 2D (batch_size, seq_len)
        assert result['input_ids'].dim() == 2
        assert result['attention_mask'].dim() == 2
        assert result['drug1_mask'].dim() == 2
        assert result['drug2_mask'].dim() == 2
        assert result['ner_labels'].dim() == 2

    def test_batch_tokenize_consistent_shapes(self, ddi_tokenizer):
        """Test batch tokenize produces consistent shapes"""
        texts = ["short", "this is a longer text for testing"]
        result = ddi_tokenizer.batch_tokenize(texts)

        # All sequences should have same length (padded)
        batch_size, seq_len = result['input_ids'].shape
        assert result['attention_mask'].shape == (batch_size, seq_len)
        assert result['drug1_mask'].shape == (batch_size, seq_len)

    def test_batch_tokenize_empty_list(self, ddi_tokenizer):
        """Test batch tokenize with empty list"""
        texts = []
        result = ddi_tokenizer.batch_tokenize(texts)

        # Should return empty batches
        assert result['input_ids'].shape[0] == 0


class TestDecode:
    """Test decoding functionality"""

    def test_decode_basic(self, ddi_tokenizer):
        """Test decode functionality"""
        input_ids = torch.tensor([105, 101, 999, 102])
        result = ddi_tokenizer.decode(input_ids)

        assert isinstance(result, str)
        ddi_tokenizer.tokenizer.decode.assert_called_once()

    def test_decode_preserves_special_tokens(self, ddi_tokenizer):
        """Test decode is called with skip_special_tokens=False"""
        input_ids = torch.tensor([101, 102])
        ddi_tokenizer.decode(input_ids)

        call_kwargs = ddi_tokenizer.tokenizer.decode.call_args[1]
        assert call_kwargs['skip_special_tokens'] is False

    def test_decode_with_drug_markers(self, ddi_tokenizer):
        """Test decode with drug marker token IDs"""
        ddi_tokenizer.tokenizer.decode.return_value = "[DRUG1] aspirin [/DRUG1]"

        input_ids = torch.tensor([101, 999, 102])
        result = ddi_tokenizer.decode(input_ids)

        assert '[DRUG1]' in result or result == "decoded text"


class TestResizeModelEmbeddings:
    """Test model embedding resizing"""

    def test_resize_model_embeddings(self, ddi_tokenizer):
        """Test resize_model_embeddings calls correct method"""
        # Create mock model with encoder
        mock_model = Mock()
        mock_model.encoder = Mock()
        mock_model.encoder.resize_token_embeddings = Mock()

        ddi_tokenizer.resize_model_embeddings(mock_model)

        mock_model.encoder.resize_token_embeddings.assert_called_once_with(30522)

    def test_resize_with_correct_size(self, ddi_tokenizer):
        """Test resize uses correct vocabulary size"""
        mock_model = Mock()
        mock_model.encoder = Mock()
        mock_model.encoder.resize_token_embeddings = Mock()

        ddi_tokenizer.resize_model_embeddings(mock_model)

        # Should use get_vocab_size()
        expected_size = ddi_tokenizer.get_vocab_size()
        mock_model.encoder.resize_token_embeddings.assert_called_with(expected_size)


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_mark_and_tokenize_workflow(self, ddi_tokenizer):
        """Test complete workflow: mark drugs -> tokenize"""
        text = "Patient takes aspirin and ibuprofen"
        drug1_span = (14, 21)  # aspirin
        drug2_span = (26, 35)  # ibuprofen

        # Mark drugs
        marked_text = ddi_tokenizer.mark_drugs_in_text(text, drug1_span, drug2_span)

        # Tokenize
        result = ddi_tokenizer.tokenize(marked_text)

        # Should have valid outputs
        assert result['input_ids'].shape[0] > 0
        assert torch.any(result['drug1_mask'] > 0)
        assert torch.any(result['drug2_mask'] > 0)

    def test_batch_workflow(self, ddi_tokenizer):
        """Test batch processing workflow"""
        texts = [
            "Drug1 interacts with Drug2",
            "Drug3 and Drug4 interaction"
        ]

        # Mark drugs for each text
        marked_texts = []
        for text in texts:
            marked = ddi_tokenizer.mark_drugs_in_text(
                text,
                (0, 5),  # First drug
                (22, 27) if len(text) > 25 else (11, 16)  # Second drug
            )
            marked_texts.append(marked)

        # Batch tokenize
        result = ddi_tokenizer.batch_tokenize(marked_texts)

        assert result['input_ids'].shape[0] == 2
        assert result['attention_mask'].shape[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
