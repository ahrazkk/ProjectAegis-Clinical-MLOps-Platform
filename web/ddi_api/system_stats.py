"""Helpers for global system counters used by dashboard analytics."""

from django.db import IntegrityError
from django.db.models import F

from .models import SystemStats


def increment_total_scans(amount: int = 1) -> int:
    """Atomically increment the global scan counter and return the latest value."""
    if amount <= 0:
        stats, _ = SystemStats.objects.get_or_create(name='global')
        return stats.total_scans

    updated = SystemStats.objects.filter(name='global').update(total_scans=F('total_scans') + amount)

    if not updated:
        try:
            SystemStats.objects.create(name='global', total_scans=amount)
            return amount
        except IntegrityError:
            SystemStats.objects.filter(name='global').update(total_scans=F('total_scans') + amount)

    return SystemStats.objects.only('total_scans').get(name='global').total_scans


def get_total_scans() -> int:
    """Return the global scan counter, creating the row if missing."""
    stats, _ = SystemStats.objects.get_or_create(name='global')
    return stats.total_scans
