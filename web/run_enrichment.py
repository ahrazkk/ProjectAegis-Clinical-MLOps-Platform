#!/usr/bin/env python
"""Script to run enrichment against Neo4j Aura cloud instance."""
import os
import sys

# Set environment variables for Neo4j Aura
os.environ['NEO4J_URI'] = 'neo4j+s://ca47aebc.databases.neo4j.io'
os.environ['NEO4J_USER'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'BYKmHWoR2DeEiiiwO6qBAET273OIaaGv1ZatYpU_vtM'

# Change to the web directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ProjectAegis.settings')
import django
django.setup()

# Now run the management command
from django.core.management import call_command

print("=" * 60)
print("🚀 RUNNING ENRICHMENT AGAINST NEO4J AURA CLOUD")
print("=" * 60)
print(f"URI: {os.environ['NEO4J_URI']}")
print()

# Run with all enrichment options
call_command('enrich_drug_data', all=True, limit=2500)
