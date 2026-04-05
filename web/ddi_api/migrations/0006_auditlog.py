from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ddi_api', '0005_systemstats_llm_tracking'),
    ]

    operations = [
        migrations.CreateModel(
            name='AuditLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('event_type', models.CharField(choices=[
                    ('prediction', 'DDI Prediction'),
                    ('poly_prediction', 'Polypharmacy Prediction'),
                    ('correction_created', 'Correction Created'),
                    ('correction_reviewed', 'Correction Reviewed'),
                    ('chat_query', 'Chat Query'),
                    ('export', 'Data Export'),
                ], db_index=True, max_length=30)),
                ('actor', models.CharField(default='user', max_length=100)),
                ('session_id', models.CharField(blank=True, max_length=100, null=True)),
                ('payload', models.JSONField(default=dict)),
                ('ip_address', models.GenericIPAddressField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True, db_index=True)),
            ],
            options={
                'ordering': ['-created_at'],
                'indexes': [
                    models.Index(fields=['event_type', 'created_at'], name='ddi_api_aud_event_t_idx'),
                ],
            },
        ),
    ]
