"""One-off script: login to Label Studio and get API token."""
import re
import requests

s = requests.Session()

# Step 1: GET main page
r = s.get("http://localhost:8080/")
print("GET / status:", r.status_code)

# Get CSRF
csrf = None
for c in s.cookies:
    if c.name == "csrftoken":
        csrf = c.value
        break
print("CSRF:", csrf[:50] if csrf else "N/A")

# Step 2: Look for embedded user/token data in page
if "LS_USER" in r.text:
    for m in re.finditer(r"LS_USER\s*=\s*({[^;]+})", r.text):
        print("LS_USER found:", m.group(1))

# Step 3: Explore API root paths
for path in ["/api/", "/api/v1/", "/api/user/", "/api/users/", "/api/me/"]:
    rp = s.get(f"http://localhost:8080{path}")
    if rp.status_code == 200:
        print(f"{path}: {rp.text[:300]}")
    else:
        print(f"{path}: {rp.status_code}")

# Step 4: Check current user via Django REST Framework session
r3 = s.get("http://localhost:8080/api/projects/")
print("GET /api/projects/ via session:", r3.status_code)
if r3.status_code == 200:
    print("Already authenticated!")
    print(r3.text[:500])
else:
    print("Not authenticated via session")
    # Try with CSRF token
    r4 = s.get(
        "http://localhost:8080/api/projects/",
        headers={"X-CSRFToken": csrf, "Referer": "http://localhost:8080/"},
    )
    print("With CSRF header:", r4.status_code)
    if r4.status_code == 200:
        print(r4.text[:500])
