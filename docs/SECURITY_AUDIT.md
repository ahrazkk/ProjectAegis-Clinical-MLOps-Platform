# Project Aegis - Security Audit Report

**Date:** January 25, 2026  
**Domain:** aegishealth.dev  
**Project:** project-aegis-485017  
**Auditor:** GitHub Copilot

---

## Executive Summary

Project Aegis is a public DDI (Drug-Drug Interaction) prediction tool with **no user authentication or sensitive data storage**. The security posture is appropriate for a public API/demo, but there are a few best-practice improvements to make before production.

**Overall Risk Level:** 🟡 Low-Medium

---

## 🔴 Issues to Fix (Priority: High)

### 1. Django Secret Key Hardcoded in Source Code

**File:** `web/ProjectAegis/settings.py` (Line 22)

**Current Code:**
```python
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'django-insecure-k5a8aa$4z4zprc2yhp86y&6sjs4pyz&%u4uo#p@@m8*nm^(_q0')
```

**Risk:** 
- The fallback secret key is visible in the source code
- If this key is used in production, attackers could forge session cookies and CSRF tokens
- Session hijacking becomes possible

**Impact:** Medium (you have no user accounts, but still bad practice)

**Fix:**
1. Generate a new secret key:
   ```bash
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```

2. Set it as a Cloud Run environment variable:
   ```bash
   gcloud run services update aegis-backend \
     --set-env-vars="DJANGO_SECRET_KEY=your-new-key-here" \
     --region us-central1 \
     --project project-aegis-485017
   ```

3. Remove the fallback from settings.py:
   ```python
   SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')
   if not SECRET_KEY:
       raise ValueError("DJANGO_SECRET_KEY environment variable is required")
   ```

---

### 2. DEBUG=True in Production

**File:** `web/ProjectAegis/settings.py` (Line 24)

**Current Code:**
```python
DEBUG = os.environ.get('DEBUG', 'True') == 'True'
```

**Risk:**
- Detailed error pages expose internal code structure, file paths, and stack traces
- Helps attackers understand your application internals
- Django's debug toolbar may expose sensitive information

**Impact:** Medium (information disclosure)

**Fix:**
1. Change the default to False:
   ```python
   DEBUG = os.environ.get('DEBUG', 'False') == 'True'
   ```

2. Or set explicitly in Cloud Run:
   ```bash
   gcloud run services update aegis-backend \
     --set-env-vars="DEBUG=False" \
     --region us-central1 \
     --project project-aegis-485017
   ```

---

### 3. CORS_ALLOW_ALL_ORIGINS = True

**File:** `web/ProjectAegis/settings.py` (Line 62)

**Current Code:**
```python
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://aegis-frontend-667446742007.us-central1.run.app",
]
CORS_ALLOW_ALL_ORIGINS = True  # Temporary for MVP
```

**Risk:**
- Any website can make requests to your API from a browser
- Potential for abuse or API scraping
- Cross-site request forgery risks

**Impact:** Low (no authentication means limited exploit potential)

**Fix:**
1. Remove `CORS_ALLOW_ALL_ORIGINS = True`
2. Add your custom domain to allowed origins:
   ```python
   CORS_ALLOWED_ORIGINS = [
       "http://localhost:5173",
       "http://127.0.0.1:5173",
       "https://aegis-frontend-667446742007.us-central1.run.app",
       "https://aegishealth.dev",
   ]
   # CORS_ALLOW_ALL_ORIGINS = True  # REMOVED
   ```

---

## 🟡 Issues to Consider (Priority: Medium)

### 4. No Rate Limiting

**Status:** Not implemented

**Risk:**
- Anyone can spam your API with unlimited requests
- Could increase Cloud Run costs significantly
- Potential denial of service

**Impact:** Cost increase, not a security breach

**Recommendation:**
Add Django REST Framework throttling:
```python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',  # 100 requests per hour per IP
    }
}
```

---

### 5. No Authentication on API

**Status:** AllowAny permissions on all endpoints

**Current Code:**
```python
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}
```

**Risk:** Anyone can use your API

**Impact:** None for your use case (public DDI tool)

**Recommendation:** Keep as-is for a public demo. Add API keys only if you need to track usage.

---

### 6. ALLOWED_HOSTS Contains Wildcard

**File:** `web/ProjectAegis/settings.py` (Line 26)

**Current Code:**
```python
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '.run.app', '*']
```

**Risk:** 
- The `*` allows any host header, which can be exploited for cache poisoning attacks
- Less relevant for Cloud Run since it handles host headers

**Fix:**
```python
ALLOWED_HOSTS = [
    'localhost', 
    '127.0.0.1', 
    '.run.app',
    'aegishealth.dev',
    'www.aegishealth.dev',
]
```

---

## 🟢 No Action Required (Already Secure)

### 7. Environment Files Not in Git ✅

**Status:** SECURE

`.gitignore` properly excludes:
- `.env`
- `**/.env`

No secrets are committed to GitHub.

---

### 8. HTTPS Enforcement ✅

**Status:** SECURE

Cloud Run automatically:
- Provides managed SSL certificates
- Redirects HTTP to HTTPS
- Uses TLS 1.2+

---

### 9. No Sensitive User Data ✅

**Status:** SECURE

The application:
- Has no user registration/login
- Stores no passwords or personal information
- Only processes drug interaction queries
- Uses a read-only drug database

---

### 10. Cloud Run Infrastructure ✅

**Status:** SECURE

Google Cloud Run provides:
- Automatic container isolation
- No SSH/shell access possible
- DDoS protection included
- Automatic security patches
- No exposed ports except 8080 (controlled)

---

### 11. Frontend Has No Secrets ✅

**Status:** SECURE

`.env.production` only contains the public API URL:
```
VITE_API_URL=https://aegis-backend-667446742007.us-central1.run.app/api/v1
```

This is not sensitive - it's the same URL anyone can see in browser network tab.

---

### 12. GitHub Repository ✅

**Status:** SECURE

Verified no sensitive files are tracked:
- No `.env` files committed
- No API keys in source code
- No passwords in code

---

## Quick Fix Commands

Run these commands to fix the high-priority issues:

```bash
# 1. Generate new Django secret key
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"

# 2. Set environment variables on Cloud Run backend
gcloud run services update aegis-backend \
  --set-env-vars="DJANGO_SECRET_KEY=<paste-new-key>,DEBUG=False" \
  --region us-central1 \
  --project project-aegis-485017

# 3. Verify the update
gcloud run services describe aegis-backend \
  --region us-central1 \
  --project project-aegis-485017 \
  --format="value(spec.template.spec.containers[0].env)"
```

---

## Summary Table

| Issue | Severity | Status | Action |
|-------|----------|--------|--------|
| Hardcoded SECRET_KEY | 🔴 High | Needs Fix | Set as env var |
| DEBUG=True in prod | 🔴 High | Needs Fix | Set DEBUG=False |
| CORS_ALLOW_ALL_ORIGINS | 🔴 High | Needs Fix | Remove, add domain |
| No Rate Limiting | 🟡 Medium | Optional | Add throttling |
| No Authentication | 🟡 Medium | Acceptable | Keep for public API |
| ALLOWED_HOSTS wildcard | 🟡 Medium | Optional | Remove * |
| .env in gitignore | 🟢 Secure | ✅ Done | None |
| HTTPS/SSL | 🟢 Secure | ✅ Done | None |
| No user data | 🟢 Secure | ✅ Done | None |
| Cloud Run hardening | 🟢 Secure | ✅ Done | None |
| GitHub repo | 🟢 Secure | ✅ Done | None |

---

## Estimated Fix Time

- **High Priority Items:** 15-20 minutes
- **Medium Priority Items:** 30 minutes (optional)

---

*This audit covers the current deployment as of January 2026. Re-audit if adding user authentication or storing sensitive data.*
