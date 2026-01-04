"""
Django Models for Project Aegis DDI API

These models store drug information, interaction history,
and user search logs for the Clinical Decision Support System.
"""

from django.db import models
from django.contrib.auth.models import User


class Drug(models.Model):
    """
    Drug entity with molecular structure information.
    Contains SMILES notation for GNN processing.
    """
    drugbank_id = models.CharField(max_length=20, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    smiles = models.TextField(blank=True, null=True, help_text="SMILES molecular notation")
    description = models.TextField(blank=True, null=True)
    
    # Molecular properties (for filtering/display)
    molecular_weight = models.FloatField(null=True, blank=True)
    molecular_formula = models.CharField(max_length=100, blank=True, null=True)
    
    # Classification
    drug_class = models.CharField(max_length=255, blank=True, null=True)
    atc_code = models.CharField(max_length=50, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['name']
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['drugbank_id']),
        ]
    
    def __str__(self):
        return f"{self.name} ({self.drugbank_id})"


class DrugTarget(models.Model):
    """
    Protein/Gene targets for drugs.
    Used in the Knowledge Graph for pathway analysis.
    """
    uniprot_id = models.CharField(max_length=20, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    gene_name = models.CharField(max_length=100, blank=True, null=True)
    organism = models.CharField(max_length=100, default="Homo sapiens")
    
    # Relationships
    drugs = models.ManyToManyField(Drug, through='DrugTargetInteraction', related_name='targets')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.uniprot_id})"


class DrugTargetInteraction(models.Model):
    """
    Many-to-many relationship between drugs and their targets.
    Includes action type (inhibitor, agonist, etc.).
    """
    ACTION_TYPES = [
        ('inhibitor', 'Inhibitor'),
        ('agonist', 'Agonist'),
        ('antagonist', 'Antagonist'),
        ('substrate', 'Substrate'),
        ('binder', 'Binder'),
        ('unknown', 'Unknown'),
    ]
    
    drug = models.ForeignKey(Drug, on_delete=models.CASCADE)
    target = models.ForeignKey(DrugTarget, on_delete=models.CASCADE)
    action_type = models.CharField(max_length=50, choices=ACTION_TYPES, default='unknown')
    source = models.CharField(max_length=100, default='DrugBank')
    
    class Meta:
        unique_together = ['drug', 'target']


class DrugDrugInteraction(models.Model):
    """
    Known drug-drug interactions from curated databases.
    Used for training data and validation.
    """
    SEVERITY_LEVELS = [
        ('minor', 'Minor'),
        ('moderate', 'Moderate'),
        ('major', 'Major'),
        ('contraindicated', 'Contraindicated'),
    ]
    
    drug_a = models.ForeignKey(Drug, on_delete=models.CASCADE, related_name='interactions_as_a')
    drug_b = models.ForeignKey(Drug, on_delete=models.CASCADE, related_name='interactions_as_b')
    
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='moderate')
    description = models.TextField(blank=True, null=True)
    mechanism = models.TextField(blank=True, null=True, help_text="Mechanism of interaction")
    
    # Affected organ systems (for body map visualization)
    affected_systems = models.JSONField(default=list, blank=True)  # ['liver', 'kidney', etc.]
    
    # Source information
    source = models.CharField(max_length=100, default='DDIExtraction2013')
    pubmed_ids = models.JSONField(default=list, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['drug_a', 'drug_b']
        indexes = [
            models.Index(fields=['drug_a', 'drug_b']),
            models.Index(fields=['severity']),
        ]
    
    def __str__(self):
        return f"{self.drug_a.name} ↔ {self.drug_b.name} ({self.severity})"


class SideEffect(models.Model):
    """
    Side effects that can result from drug interactions.
    Mapped to organ systems for body visualization.
    """
    ORGAN_SYSTEMS = [
        ('brain', 'Brain/Nervous System'),
        ('heart', 'Cardiovascular'),
        ('liver', 'Hepatic'),
        ('kidney', 'Renal'),
        ('lungs', 'Respiratory'),
        ('gi', 'Gastrointestinal'),
        ('blood', 'Hematologic'),
        ('skin', 'Dermatologic'),
        ('musculoskeletal', 'Musculoskeletal'),
        ('endocrine', 'Endocrine'),
        ('other', 'Other'),
    ]
    
    umls_id = models.CharField(max_length=20, unique=True, db_index=True)
    name = models.CharField(max_length=255)
    organ_system = models.CharField(max_length=50, choices=ORGAN_SYSTEMS, default='other')
    severity_weight = models.FloatField(default=0.5, help_text="0.0 (minor) to 1.0 (fatal)")
    
    def __str__(self):
        return f"{self.name} ({self.organ_system})"


class PredictionLog(models.Model):
    """
    Log of DDI predictions made by the system.
    Used for analytics and model improvement.
    """
    # Input drugs (stored as JSON for N-way interactions)
    drug_list = models.JSONField(default=list)  # List of drug IDs or names
    
    # Prediction results
    risk_score = models.FloatField()
    calibrated_score = models.FloatField(null=True, blank=True)
    severity_prediction = models.CharField(max_length=20, blank=True)
    
    # Metadata
    model_version = models.CharField(max_length=50, default='v1.0')
    inference_time_ms = models.FloatField(null=True, blank=True)
    
    # User tracking (optional)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
        ]
    
    def __str__(self):
        drugs = ', '.join(self.drug_list[:3])
        return f"Prediction: {drugs}... (Risk: {self.risk_score:.2f})"
