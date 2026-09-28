from core.universe import load
from collections import Counter

funds = load()
print("Total funds:", len(funds))
print()
print("=== By appetite_bucket ===")
buckets = Counter(f.get("appetite_bucket") for f in funds)
for b, n in buckets.most_common():
    print("  %s: %d" % (b, n))
print()
print("=== By risk_band ===")
bands = Counter(f.get("risk_band") for f in funds)
for b, n in bands.most_common():
    print("  %s: %d" % (b, n))
print()
print("=== By risk_band_source ===")
srcs = Counter(f.get("risk_band_source") for f in funds)
for s, n in srcs.most_common():
    print("  %s: %d" % (s, n))
print()
print("=== expense_ratio_source ===")
esrcs = Counter(f.get("expense_ratio_source") for f in funds)
for s, n in esrcs.most_common():
    print("  %s: %d" % (s, n))
print()
print("=== ISSUE CHECK: liquid-named funds in wrong bucket ===")
found_issue = False
for f in funds:
    name = f.get("fund_name", "").lower()
    bucket = f.get("appetite_bucket")
    if "liquid" in name and bucket != "low":
        print("  WRONG BUCKET: %d %s -> bucket=%s risk=%s" % (
            f["scheme_code"], f["fund_name"], bucket, f["risk_band"]))
        found_issue = True
if not found_issue:
    print("  None found")

print()
print("=== ISSUE CHECK: any fund with None risk_band ===")
none_band = [f for f in funds if not f.get("risk_band")]
for f in none_band:
    print("  NO BAND: %d %s cat=%s" % (f["scheme_code"], f["fund_name"], f.get("category","")))
if not none_band:
    print("  None found")

print()
print("=== Funds per SEBI category (full category string) ===")
cats = Counter(f.get("category","unknown") for f in funds)
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print("  %d  %s" % (n, c))
