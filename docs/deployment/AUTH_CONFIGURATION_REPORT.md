
# AUTHENTICATION CONFIGURATION INVESTIGATION REPORT
## Server: jakebot@10.79.85.35:~/Jakes-agent  
## Date: 2026-02-27 13:26:48

---

## EXECUTIVE SUMMARY

Authentication in Dish-Chat is currently **DISABLED** for local development.
The system uses a **dual-mode authentication strategy**:

1. **Production Mode**: OAuth2-Proxy with X-Auth-Request-Email header (Okta SSO)
2. **Development Mode**: Authentication bypass with default test user

---

## CURRENT AUTHENTICATION STATUS

### Active Configuration

| Setting | Current Value | Location |
|---------|---------------|----------|
| **AUTH_DISABLED** | `False` | `app/config.py` line 22 |
| **LOCAL** | `True` | `app/config.py` line 20 |
| **DEFAULT_USER_EMAIL** | `"test.user@dish.com"` | `app/config.py` line 23 |
| **DEBUG** | `False` | `app/config.py` line 19 |

### Effective Behavior

✅ **Current Mode: LOCAL DEVELOPMENT (Auth Bypassed)**

Because `LOCAL = True`, the `LocalIdInjectMiddleware` is active, which:
- Automatically injects header: `X-Auth-Request-Email: test.test@dish.com`
- This simulates an authenticated user without requiring real OAuth

---

## AUTHENTICATION ARCHITECTURE

### 1. Core Authentication Logic

**File**: `app/core/user.py`

The main authentication function follows this priority:

1. If X-Auth-Request-Email header present → use it (real authenticated user)
2. If AUTH_DISABLED=True and no header → use DEFAULT_USER_EMAIL  
3. Otherwise → raise 401 Unauthorized

### 2. Local Development Middleware

**File**: `app/middlewares.py`

**Class**: `LocalIdInjectMiddleware`
- Automatically injects: `X-Auth-Request-Email: test.test@dish.com`
- **Activation**: Only when `settings.LOCAL = True` (line 42 of `app/main.py`)

### 3. FastAPI Dependency Injection

**File**: `app/dependencies.py`

```python
UserEmailDep = Annotated[str, Depends(get_user_email)]
```

This dependency is injected into protected endpoints, ensuring authentication happens before route handlers execute.

### 4. Protected Endpoints

Authentication is enforced on all endpoints using `UserEmailDep`:

- **Chat API** (`app/chat/router.py`)
- **Vault API** (`app/vault/router.py`)
- **Logs API** (`app/logs/router.py`)
- **Releases API** (`app/releases/router.py`)
- **Chat Groups API** (`app/chat_group/router.py`)
- **Visualization API** (`app/viz_router.py`)
- **Analytics API** (`app/analytics/router.py`)

---

## HOW TO ENABLE AUTHENTICATION

### Option 1: Enable Full Authentication (Production-like)

**Method**: Disable both LOCAL mode and AUTH bypass

**Changes Required**:

1. **Edit `app/config.py`** (lines 19-22):
   ```python
   # Runtime settings
   DEBUG: bool = False
   LOCAL: bool = False  # CHANGE FROM True → False
   
   # Authentication Settings
   AUTH_DISABLED: bool = False  # KEEP as False
   DEFAULT_USER_EMAIL: str = "test.user@dish.com"
   ```

2. **Alternative**: Set via environment variables in `.env`:
   ```bash
   LOCAL=false
   AUTH_DISABLED=false
   ```

**Effect**:
- ✅ Disables `LocalIdInjectMiddleware` (no auto-injected headers)
- ✅ Requires real `X-Auth-Request-Email` header on all requests
- ❌ API calls without header → 401 Unauthorized

**Testing**:
```bash
# Without header - should fail
curl http://10.79.85.35:8000/rest/api/v1/chats
# Returns: 401 Unauthorized

# With header - should succeed
curl -H "X-Auth-Request-Email: user@dish.com" \
  http://10.79.85.35:8000/rest/api/v1/chats
# Returns: Chat list
```

---

### Option 2: Keep Development Mode with Custom User

**Method**: Keep LOCAL mode but change the test user

**Changes Required**:

1. **Edit `app/middlewares.py`** (line 13):
   ```python
   headers[b"x-auth-request-email"] = b"your.name@dish.com"  # Change user
   ```

**Effect**:
- ✅ All requests automatically authenticated as specified user
- ✅ Still convenient for development
- ✅ Can test user-specific features with different emails

---

### Option 3: Bypass Authentication Entirely (Testing Only)

**Method**: Set `AUTH_DISABLED = True`

**Changes Required**:

1. **Edit `app/config.py`** (line 22):
   ```python
   AUTH_DISABLED: bool = True  # CHANGE FROM False → True
   ```

2. **Alternative**: Set via environment variable:
   ```bash
   echo "AUTH_DISABLED=true" >> .env
   ```

**Effect**:
- ✅ No headers required at all
- ✅ All requests treated as `DEFAULT_USER_EMAIL`
- ⚠️  Less realistic for testing user-specific features
- ⚠️  Should NEVER be used in production

---

## PRODUCTION DEPLOYMENT WITH OAUTH2-PROXY

### Architecture Overview

```
[User Browser]
     ↓
[OAuth2-Proxy] ← Okta SSO Authentication
     ↓ (injects X-Auth-Request-Email header)
[Dish-Chat Backend]
     ↓
[get_user_email() reads header]
     ↓
[User-specific data access]
```

### Required Setup (External to this app)

1. **Deploy OAuth2-Proxy** as reverse proxy
2. **Configure Okta SSO** integration
3. **Set OAuth2-Proxy** to inject `X-Auth-Request-Email` header
4. **Set `LOCAL=False`** in Dish-Chat config
5. **Route all traffic** through OAuth2-Proxy

---

## CONFIGURATION FILE LOCATIONS

### Primary Configuration
- **File**: `app/config.py`
- **Lines**: 19-23 (AUTH settings)
- **Class**: `Settings(BaseSettings)`

### Environment Overrides
- **Priority 1**: `.env.local` (not currently used for auth)
- **Priority 2**: `.env` (not currently used for auth)
- **Priority 3**: System environment variables

### Current .env Files
```bash
# .env (current content)
SENTRY_AUTH_TOKEN=<REDACTED_ROTATED_SENTRY_TOKEN>
# No AUTH_DISABLED or LOCAL overrides

# .env.local (current content)  
SENTRY_AUTH_TOKEN=<REDACTED_ROTATED_SENTRY_TOKEN>
# No AUTH_DISABLED or LOCAL overrides

# .env.production (current content)
SENTRY_AUTH_TOKEN=<REDACTED_ROTATED_SENTRY_TOKEN>
# No AUTH_DISABLED or LOCAL overrides
```

**Conclusion**: All auth settings use defaults from `app/config.py`

---

## RECOMMENDED APPROACH FOR YOUR SITUATION

### If You Want to Test with Real Authentication:

**Step 1**: Edit `app/config.py`
```python
LOCAL: bool = False  # Line 20
```

**Step 2**: Restart the service
```bash
ssh jakebot@10.79.85.35
cd ~/Jakes-agent
bash restart-dishchat.sh
```

**Step 3**: Test with curl
```bash
# Should fail (401)
curl http://10.79.85.35:8000/rest/api/v1/chats

# Should succeed
curl -H "X-Auth-Request-Email: your.email@dish.com" \
  http://10.79.85.35:8000/rest/api/v1/chats
```

### If You Want to Keep Development Mode:

**No changes needed!** Current setup is ideal for local development:
- All requests automatically authenticated
- No need to manage headers
- Easy to test features quickly

---

## SECURITY CONSIDERATIONS

### Current Status: ⚠️  DEVELOPMENT MODE

✅ **Safe for**:
- Local development on internal networks (10.79.x.x)
- Testing and debugging
- Internal team access

❌ **NOT safe for**:
- Public internet exposure
- Production deployments
- Multi-tenant scenarios

### To Secure:

1. ✅ Set `LOCAL=False`
2. ✅ Deploy behind OAuth2-Proxy
3. ✅ Restrict network access (firewall rules)
4. ✅ Use HTTPS/TLS in production
5. ✅ Set strong `MASTER_KEY` (for vault encryption)

---

## FILES TO MODIFY (MANUAL CHANGES)

### Primary File
**`~/Jakes-agent/app/config.py`**
```python
Line 20: LOCAL: bool = True  → False
Line 22: AUTH_DISABLED: bool = False  (keep as is)
```

### Alternative: Environment Variable
**`~/Jakes-agent/.env`**
```bash
# Add these lines:
LOCAL=false
AUTH_DISABLED=false
```

### After Changes
```bash
# Restart service
cd ~/Jakes-agent
bash restart-dishchat.sh

# Verify settings took effect
tail -f logs/backend.log
# Look for startup messages
```

---

## VERIFICATION COMMANDS

### Check Current Behavior
```bash
# Should succeed (middleware injects header)
curl http://10.79.85.35:8000/rest/api/v1/health

# Should also succeed (returns chats)
curl http://10.79.85.35:8000/rest/api/v1/chats | jq '.docs | length'
```

### After Enabling Auth
```bash
# Should fail with 401
curl http://10.79.85.35:8000/rest/api/v1/chats

# Should succeed with header
curl -H "X-Auth-Request-Email: test@dish.com" \
  http://10.79.85.35:8000/rest/api/v1/chats
```

---

## SUMMARY TABLE

| Setting | File | Line | Current | To Enable Auth |
|---------|------|------|---------|----------------|
| `LOCAL` | `app/config.py` | 20 | `True` | Change to `False` |
| `AUTH_DISABLED` | `app/config.py` | 22 | `False` | Keep as `False` |

**One-line change** to enable authentication: Set `LOCAL = False` in `app/config.py` line 20.

**Alternative**: Add `LOCAL=false` to `.env` file (no code edit needed).

---

## KEY FILES REFERENCE

1. **`app/config.py`** - Main configuration (lines 19-23)
2. **`app/core/user.py`** - Authentication logic
3. **`app/middlewares.py`** - LocalIdInjectMiddleware (line 11-15)
4. **`app/main.py`** - Middleware registration (line 42)
5. **`app/dependencies.py`** - UserEmailDep definition

---

## NEXT STEPS

1. **Decide**: Development mode (current) vs. Production mode (auth enabled)
2. **If enabling auth**: Edit `app/config.py` line 20 or add `LOCAL=false` to `.env`
3. **Restart service**: `bash restart-dishchat.sh`
4. **Test**: Verify 401 errors for requests without headers
5. **Optional**: Set up OAuth2-Proxy for SSO integration

---

*Investigation complete - no changes made per user request*
*Ready for manual configuration when needed*
