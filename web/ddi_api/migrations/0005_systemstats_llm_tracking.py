from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ddi_api', '0004_systemstats'),
    ]

    operations = [
        migrations.AddField(
            model_name='systemstats',
            name='llm_input_tokens',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='systemstats',
            name='llm_output_tokens',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='systemstats',
            name='llm_queries',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='systemstats',
            name='llm_cost_usd',
            field=models.FloatField(default=0.0),
        ),
    ]
