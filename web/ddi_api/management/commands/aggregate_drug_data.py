"""
Aggregate Drug Data from Multiple Sources

This command uses the unified data aggregator to fetch and merge
drug information from:
- RxNorm (NIH) - Names, NDC codes, interactions
- OpenFDA - Labels, adverse events
- PubChem - Molecular structures, properties
- DrugBank - Pharmacology, targets (if XML available)

Usage:
    # Fetch data for specific drugs
    python manage.py aggregate_drug_data --drugs "aspirin" "metformin" "lisinopril"
    
    # Fetch from a file (one drug per line)
    python manage.py aggregate_drug_data --file drugs.txt
    
    # Fetch all drugs currently in database
    python manage.py aggregate_drug_data --all --limit 100
    
    # Export to file
    python manage.py aggregate_drug_data --drugs "aspirin" --export drugs.json
    
    # Include adverse event data (slower)
    python manage.py aggregate_drug_data --drugs "aspirin" --adverse-events
"""

import json
import logging
from typing import List, Optional
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ddi_api.models import Drug
from ddi_api.services.data_sources import (
    RxNormClient,
    OpenFDAClient,
    PubChemClient,
    DrugDataAggregator,
    create_aggregator
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Aggregate drug data from multiple pharmaceutical databases'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--drugs',
            nargs='+',
            help='List of drug names to fetch'
        )
        parser.add_argument(
            '--file',
            type=str,
            help='File containing drug names (one per line)'
        )
        parser.add_argument(
            '--all',
            action='store_true',
            help='Fetch data for all drugs in database'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=None,
            help='Limit number of drugs to process'
        )
        parser.add_argument(
            '--export',
            type=str,
            help='Export results to JSON file'
        )
        parser.add_argument(
            '--export-csv',
            type=str,
            help='Export results to CSV file'
        )
        parser.add_argument(
            '--adverse-events',
            action='store_true',
            help='Include adverse event data (slower)'
        )
        parser.add_argument(
            '--no-interactions',
            action='store_true',
            help='Skip fetching interactions'
        )
        parser.add_argument(
            '--update-db',
            action='store_true',
            help='Update database with fetched data'
        )
        parser.add_argument(
            '--drugbank-xml',
            type=str,
            help='Path to DrugBank XML file (optional)'
        )
        parser.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of parallel workers for fetching'
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes'
        )
    
    def handle(self, *args, **options):
        self.verbosity = options.get('verbosity', 1)
        
        # Get list of drugs to process
        drug_names = self._get_drug_names(options)
        
        if not drug_names:
            raise CommandError(
                'No drugs specified. Use --drugs, --file, or --all'
            )
        
        if options['limit']:
            drug_names = drug_names[:options['limit']]
        
        self.stdout.write(
            f"Processing {len(drug_names)} drugs..."
        )
        
        if options['dry_run']:
            self.stdout.write("DRY RUN - showing drugs to process:")
            for name in drug_names[:20]:
                self.stdout.write(f"  - {name}")
            if len(drug_names) > 20:
                self.stdout.write(f"  ... and {len(drug_names) - 20} more")
            return
        
        # Create aggregator
        aggregator = self._create_aggregator(options)
        
        # Fetch data
        drugs = aggregator.fetch_drugs(
            drug_names,
            max_workers=options['workers'],
            include_interactions=not options['no_interactions'],
            include_adverse_events=options['adverse_events']
        )
        
        self.stdout.write(
            self.style.SUCCESS(f"Successfully fetched {len(drugs)} drugs")
        )
        
        # Show summary
        self._print_summary(drugs)
        
        # Export if requested
        if options['export']:
            aggregator.export_to_json(options['export'], drugs)
            self.stdout.write(f"Exported to {options['export']}")
        
        if options['export_csv']:
            aggregator.export_to_csv(options['export_csv'], drugs)
            self.stdout.write(f"Exported to {options['export_csv']}")
        
        # Update database if requested
        if options['update_db']:
            self._update_database(drugs)
    
    def _get_drug_names(self, options) -> List[str]:
        """Get list of drug names from various sources."""
        drug_names = []
        
        # From command line
        if options['drugs']:
            drug_names.extend(options['drugs'])
        
        # From file
        if options['file']:
            file_path = Path(options['file'])
            if not file_path.exists():
                raise CommandError(f"File not found: {options['file']}")
            
            with open(file_path) as f:
                for line in f:
                    name = line.strip()
                    if name and not name.startswith('#'):
                        drug_names.append(name)
        
        # From database
        if options['all']:
            db_drugs = Drug.objects.values_list('name', flat=True)
            drug_names.extend(db_drugs)
        
        # Deduplicate while preserving order
        seen = set()
        unique_names = []
        for name in drug_names:
            name_lower = name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                unique_names.append(name)
        
        return unique_names
    
    def _create_aggregator(self, options) -> DrugDataAggregator:
        """Create the data aggregator with configured clients."""
        # Check for DrugBank XML
        drugbank = None
        if options['drugbank_xml']:
            xml_path = Path(options['drugbank_xml'])
            if xml_path.exists():
                from ddi_api.services.data_sources import DrugBankParser
                drugbank = DrugBankParser(str(xml_path))
                self.stdout.write("Using DrugBank XML file")
            else:
                self.stderr.write(
                    f"Warning: DrugBank XML not found at {xml_path}"
                )
        
        return DrugDataAggregator(
            rxnorm=RxNormClient(),
            openfda=OpenFDAClient(),
            pubchem=PubChemClient(),
            drugbank=drugbank
        )
    
    def _print_summary(self, drugs):
        """Print summary of fetched data."""
        if not drugs:
            return
        
        # Count by source
        source_counts = {}
        for drug in drugs:
            for source in drug.sources:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        self.stdout.write("\nData sources used:")
        for source, count in sorted(source_counts.items()):
            self.stdout.write(f"  {source}: {count} drugs")
        
        # Count with SMILES
        with_smiles = sum(1 for d in drugs if d.smiles)
        self.stdout.write(f"\nDrugs with SMILES: {with_smiles}/{len(drugs)}")
        
        # Count with interactions
        with_interactions = sum(1 for d in drugs if d.drug_interactions)
        total_interactions = sum(len(d.drug_interactions) for d in drugs)
        self.stdout.write(
            f"Drugs with interactions: {with_interactions}/{len(drugs)} "
            f"({total_interactions} total)"
        )
    
    @transaction.atomic
    def _update_database(self, drugs):
        """Update database with fetched drug data."""
        updated = 0
        created = 0
        interactions_count = 0
        
        # First pass: Create/Update all drugs
        drug_map = {} # Cache for interaction linking
        
        for unified_drug in drugs:
            try:
                # Use drugbank_id as primary key if available, else name
                defaults = {
                    'name': unified_drug.name,
                    'molecular_formula': unified_drug.molecular_formula,
                    'molecular_weight': unified_drug.molecular_weight,
                    'smiles': unified_drug.smiles,
                    'description': unified_drug.description,
                    'drug_class': (
                        unified_drug.therapeutic_classes[0] 
                        if unified_drug.therapeutic_classes else None
                    ),
                }
                
                # Handle lookup - prioritize drugbank_id if we have it
                if unified_drug.drugbank_id:
                    drug, was_created = Drug.objects.update_or_create(
                        drugbank_id=unified_drug.drugbank_id,
                        defaults=defaults
                    )
                else:
                    # Fallback to name-based lookup (update if exists)
                    drug, was_created = Drug.objects.update_or_create(
                        name__iexact=unified_drug.name,
                        defaults={**defaults, 'drugbank_id': f"TEMP_{unified_drug.name[:10]}"}
                    )
                
                drug_map[unified_drug.name.lower()] = drug
                
                if was_created:
                    created += 1
                else:
                    updated += 1
                    
            except Exception as e:
                logger.warning(f"Failed to update drug {unified_drug.name}: {e}")

        # Second pass: Link interactions
        from ddi_api.models import DrugDrugInteraction
        
        for unified_drug in drugs:
            if not unified_drug.drug_interactions:
                continue
                
            drug_a = drug_map.get(unified_drug.name.lower())
            if not drug_a:
                continue
                
            for interaction in unified_drug.drug_interactions:
                other_name = interaction.get('drug_name')
                if not other_name:
                    continue
                    
                # Try to find the other drug in our map or DB
                drug_b = drug_map.get(other_name.lower())
                if not drug_b:
                    # Try DB lookup by name
                    drug_b = Drug.objects.filter(name__iexact=other_name).first()
                
                # If drug_b exists and isn't drug_a
                if drug_b and drug_b != drug_a:
                    try:
                        # Ensure A < B for uniqueness (simple lexical sort)
                        if drug_a.id > drug_b.id:
                            d1, d2 = drug_b, drug_a
                        else:
                            d1, d2 = drug_a, drug_b
                            
                        # Map severity text to choices
                        severity_text = interaction.get('severity', 'moderate').lower()
                        valid_severities = ['minor', 'moderate', 'major', 'contraindicated']
                        severity = severity_text if severity_text in valid_severities else 'moderate'
                        
                        DrugDrugInteraction.objects.update_or_create(
                            drug_a=d1,
                            drug_b=d2,
                            defaults={
                                'severity': severity,
                                'description': interaction.get('description', ''),
                                'source': interaction.get('source', 'Aggregator')
                            }
                        )
                        interactions_count += 1
                    except Exception as e:
                        # Ignore duplicates or validation errors
                        pass

        self.stdout.write(
            self.style.SUCCESS(
                f"Database updated: {created} created, {updated} updated, "
                f"{interactions_count} interactions linked"
            )
        )
