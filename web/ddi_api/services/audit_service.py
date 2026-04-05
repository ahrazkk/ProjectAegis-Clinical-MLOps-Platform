"""
Audit Service for Project Aegis
Logs all significant events for compliance and debugging.
"""
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from django.utils import timezone

logger = logging.getLogger(__name__)


class AuditService:
    @staticmethod
    def log_event(event_type: str, payload: Dict[str, Any] = None,
                  actor: str = 'user', session_id: str = None,
                  ip_address: str = None):
        """Log an audit event. Fire-and-forget, never raises."""
        try:
            from ..models import AuditLog
            AuditLog.objects.create(
                event_type=event_type,
                actor=actor,
                session_id=session_id or '',
                payload=payload or {},
                ip_address=ip_address,
            )
        except Exception as e:
            logger.warning("Audit log failed: %s", e)

    @staticmethod
    def get_trail(event_type: str = None, limit: int = 50,
                  since_hours: int = None) -> List[Dict]:
        """Query audit trail."""
        try:
            from ..models import AuditLog
            qs = AuditLog.objects.all()
            if event_type:
                qs = qs.filter(event_type=event_type)
            if since_hours:
                cutoff = timezone.now() - timedelta(hours=since_hours)
                qs = qs.filter(created_at__gte=cutoff)
            return list(qs.values('id', 'event_type', 'actor', 'session_id',
                                  'payload', 'ip_address', 'created_at')[:limit])
        except Exception as e:
            logger.warning("Audit query failed: %s", e)
            return []

    @staticmethod
    def get_summary() -> Dict:
        """Get event count summary."""
        try:
            from ..models import AuditLog
            from django.db.models import Count
            counts = dict(
                AuditLog.objects.values_list('event_type')
                .annotate(c=Count('id'))
                .values_list('event_type', 'c')
            )
            total = AuditLog.objects.count()
            return {'total': total, 'by_type': counts}
        except Exception:
            return {'total': 0, 'by_type': {}}
