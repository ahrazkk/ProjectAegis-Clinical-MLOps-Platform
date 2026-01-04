"""
API Serializers for Project Aegis DDI API

These serializers handle request/response validation and formatting
for the REST API endpoints.
"""

from rest_framework import serializers
from .models import Drug, DrugTarget, DrugDrugInteraction, SideEffect, PredictionLog


# ============== Request Serializers ==============

class DrugInputSerializer(serializers.Serializer):
    """Input for a single drug in DDI check."""
    name = serializers.CharField(max_length=255)
    smiles = serializers.CharField(required=False, allow_blank=True)
    drugbank_id = serializers.CharField(max_length=20, required=False)


class DDIPredictionRequestSerializer(serializers.Serializer):
    """Request body for DDI prediction endpoint."""
    drug_a = DrugInputSerializer()
    drug_b = DrugInputSerializer()
    include_explanation = serializers.BooleanField(default=True)
    include_alternatives = serializers.BooleanField(default=False)


class PolypharmacyRequestSerializer(serializers.Serializer):
    """Request body for N-way drug interaction check."""
    drugs = DrugInputSerializer(many=True)
    
    def validate_drugs(self, value):
        if len(value) < 2:
            raise serializers.ValidationError("At least 2 drugs required for interaction check.")
        if len(value) > 20:
            raise serializers.ValidationError("Maximum 20 drugs supported for polypharmacy check.")
        return value


class ChatRequestSerializer(serializers.Serializer):
    """Request body for the GraphRAG chatbot."""
    message = serializers.CharField(max_length=2000)
    context_drugs = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=list
    )
    session_id = serializers.CharField(required=False, allow_blank=True, allow_null=True)


# ============== Response Serializers ==============

class AffectedSystemSerializer(serializers.Serializer):
    """Organ system affected by an interaction."""
    system = serializers.CharField()
    severity = serializers.FloatField(min_value=0, max_value=1)
    symptoms = serializers.ListField(child=serializers.CharField(), required=False)


class DDIPredictionResponseSerializer(serializers.Serializer):
    """Response from DDI prediction endpoint."""
    drug_a = serializers.CharField()
    drug_b = serializers.CharField()
    
    # Risk assessment
    risk_score = serializers.FloatField(min_value=0, max_value=1)
    risk_level = serializers.ChoiceField(choices=['low', 'medium', 'high', 'critical'])
    severity = serializers.ChoiceField(choices=['no_interaction', 'minor', 'moderate', 'major'])
    confidence = serializers.FloatField(min_value=0, max_value=1)
    
    # Explanation
    mechanism_hypothesis = serializers.CharField()
    affected_systems = AffectedSystemSerializer(many=True)
    
    # Molecular explanation (for XAI visualization)
    explanation = serializers.DictField(required=False)
    
    # Alternatives (if requested)
    safer_alternatives = serializers.ListField(
        child=serializers.DictField(),
        required=False
    )


class InteractionEdgeSerializer(serializers.Serializer):
    """Edge in the polypharmacy network graph."""
    source = serializers.CharField()
    target = serializers.CharField()
    risk_score = serializers.FloatField()
    severity = serializers.CharField()
    affected_systems = serializers.ListField(child=serializers.CharField())


class PolypharmacyResponseSerializer(serializers.Serializer):
    """Response from polypharmacy endpoint."""
    drugs = serializers.ListField(child=serializers.CharField())
    interactions = InteractionEdgeSerializer(many=True)
    
    # Summary statistics
    total_interactions = serializers.IntegerField()
    max_risk_score = serializers.FloatField()
    overall_risk_level = serializers.ChoiceField(choices=['low', 'medium', 'high', 'critical'])
    
    # Hub analysis
    hub_drug = serializers.CharField(allow_null=True)
    hub_interaction_count = serializers.IntegerField()
    
    # Body map data
    body_map = serializers.DictField(required=False)


class ChatResponseSerializer(serializers.Serializer):
    """Response from GraphRAG chatbot."""
    response = serializers.CharField()
    sources = serializers.ListField(
        child=serializers.DictField(),
        required=False
    )
    related_drugs = serializers.ListField(
        child=serializers.CharField(),
        required=False
    )
    session_id = serializers.CharField()


# ============== Model Serializers ==============

class DrugSerializer(serializers.ModelSerializer):
    """Serializer for Drug model."""
    
    class Meta:
        model = Drug
        fields = [
            'id', 'drugbank_id', 'name', 'smiles', 'description',
            'molecular_weight', 'molecular_formula', 'drug_class', 'atc_code'
        ]


class DrugTargetSerializer(serializers.ModelSerializer):
    """Serializer for DrugTarget model."""
    
    class Meta:
        model = DrugTarget
        fields = ['id', 'uniprot_id', 'name', 'gene_name', 'organism']


class DrugDrugInteractionSerializer(serializers.ModelSerializer):
    """Serializer for DDI records."""
    drug_a_name = serializers.CharField(source='drug_a.name', read_only=True)
    drug_b_name = serializers.CharField(source='drug_b.name', read_only=True)
    
    class Meta:
        model = DrugDrugInteraction
        fields = [
            'id', 'drug_a', 'drug_a_name', 'drug_b', 'drug_b_name',
            'severity', 'description', 'mechanism', 'affected_systems',
            'source', 'pubmed_ids'
        ]


class SideEffectSerializer(serializers.ModelSerializer):
    """Serializer for SideEffect model."""
    
    class Meta:
        model = SideEffect
        fields = ['id', 'umls_id', 'name', 'organ_system', 'severity_weight']


class PredictionLogSerializer(serializers.ModelSerializer):
    """Serializer for prediction history."""
    
    class Meta:
        model = PredictionLog
        fields = [
            'id', 'drug_list', 'risk_score', 'calibrated_score',
            'severity_prediction', 'model_version', 'inference_time_ms',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']
