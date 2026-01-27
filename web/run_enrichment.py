#!/usr/bin/env python
"""Script to run enrichment against Neo4j Aura cloud instance."""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify required environment variables are set
required_vars = ['NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD']
for var in required_vars:
    if not os.environ.get(var):
        print(f"ERROR: {var} environment variable is not set.")
        print("Please set it in your .env file or environment.")
        sys.exit(1)

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
