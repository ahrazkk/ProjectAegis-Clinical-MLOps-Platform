from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ddi_api', '0002_drug_brand_names_drug_dosage_form_drug_generic_name_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictionlog',
            name='provenance',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='predictionlog',
            name='raw_score',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
