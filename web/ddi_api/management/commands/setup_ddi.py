"""
Django management command: Setup DDI Knowledge Graph and Train Model
Usage:
    python manage.py setup_ddi --ingest --train
    python manage.py setup_ddi --check
"""

from django.core.management.base import BaseCommand, CommandError
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Setup DDI Knowledge Graph: ingest data and train model'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--check',
            action='store_true',
            help='Check Neo4j connection status'
        )
        parser.add_argument(
            '--ingest',
            action='store_true',
            help='Ingest sample drug data into Neo4j'
        )
        parser.add_argument(
            '--train',
            action='store_true',
            help='Train the DDI prediction model'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of training epochs (default: 50)'
        )
        parser.add_argument(
            '--full-dataset',
            action='store_true',
            help='Download and use full TWOSIDES dataset instead of sample data'
        )
    
    def handle(self, *args, **options):
        from ddi_api.services.knowledge_graph import KnowledgeGraphService
        from ddi_api.services.data_ingestion import DrugDataIngestion
        from ddi_api.services.train_model import run_training
        
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(self.style.SUCCESS('DDI System Setup'))
        self.stdout.write(self.style.SUCCESS('=' * 50))
        
        # Check connection
        if options['check'] or options['ingest']:
            self.stdout.write('\nChecking Neo4j connection...')
            if KnowledgeGraphService.is_connected():
                self.stdout.write(self.style.SUCCESS('✓ Neo4j is connected!'))
                
                # Show stats
                stats = KnowledgeGraphService.get_stats()
                self.stdout.write(f"  Drugs: {stats.get('drug_count', 0)}")
                self.stdout.write(f"  Targets: {stats.get('target_count', 0)}")
                self.stdout.write(f"  Interactions: {stats.get('interaction_count', 0)}")
            else:
                self.stdout.write(self.style.ERROR('✗ Cannot connect to Neo4j'))
                self.stdout.write('  Make sure Neo4j is running at bolt://localhost:7687')
                
                if options['ingest']:
                    raise CommandError('Neo4j connection required for data ingestion')
        
        # Ingest data
        if options['ingest']:
            self.stdout.write('\nIngesting drug data...')
            ingestion = DrugDataIngestion()
            
            success = ingestion.run_full_ingestion(
                use_sample=not options['full_dataset']
            )
            
            if success:
                self.stdout.write(self.style.SUCCESS('✓ Data ingestion complete!'))
            else:
                raise CommandError('Data ingestion failed')
        
        # Train model
        if options['train']:
            self.stdout.write('\nTraining DDI prediction model...')
            self.stdout.write(f'  Epochs: {options["epochs"]}')
            
            try:
                trainer = run_training(epochs=options['epochs'])
                if trainer:
                    self.stdout.write(self.style.SUCCESS('✓ Model training complete!'))
                    self.stdout.write(f"  Best AUC: {max(trainer.history['val_auc']):.4f}")
                else:
                    self.stdout.write(self.style.WARNING('⚠ Training completed with issues'))
            except Exception as e:
                raise CommandError(f'Training failed: {e}')
        
        self.stdout.write(self.style.SUCCESS('\n' + '=' * 50))
        self.stdout.write(self.style.SUCCESS('Setup complete!'))
