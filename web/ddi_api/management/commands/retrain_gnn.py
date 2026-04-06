"""
Django management command to trigger GNN retraining from approved corrections.

Usage:
    python manage.py retrain_gnn
    python manage.py retrain_gnn --epochs 100 --lr 0.0005
    python manage.py retrain_gnn --batch-size 128 --patience 15
"""

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Retrain the GNN model using approved corrections from Neo4j'

    def add_arguments(self, parser):
        parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
        parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
        parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
        parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')

    def handle(self, *args, **options):
        from ddi_api.services.retraining_pipeline import get_pipeline

        config = {
            'epochs': options['epochs'],
            'learning_rate': options['lr'],
            'batch_size': options['batch_size'],
            'patience': options['patience'],
        }

        self.stdout.write(self.style.NOTICE(f'Starting GNN retraining with config: {config}'))

        pipeline = get_pipeline()
        result = pipeline.run_full_pipeline(config=config)

        if result.get('status') == 'completed':
            self.stdout.write(self.style.SUCCESS(
                f"Retraining completed successfully!\n"
                f"  Corrections used: {result.get('correction_count', 0)}\n"
                f"  Checkpoint: {result.get('checkpoint_path', 'N/A')}\n"
                f"  Metrics: {result.get('test_metrics', {})}"
            ))
        else:
            self.stdout.write(self.style.ERROR(
                f"Retraining failed: {result.get('error', 'Unknown error')}"
            ))
