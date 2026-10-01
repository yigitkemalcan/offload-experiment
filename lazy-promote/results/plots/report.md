# Lazy-promotion results

```json
{
  "trials": 200,
  "expected_calls": 4487,
  "paired_calls": 4487,
  "compared_calls": 4487,
  "issues": [],
  "behavior_differences_between_modes": 0,
  "original_behavior_differences": 5,
  "original_returncode_unavailable": 83,
  "median_slowdown": 1.4124452646049648,
  "p99_slowdown": 2.6449457295197885,
  "max_slowdown": 3.48534035044869,
  "overall_slowdown": 1.399448281420706,
  "residual_user_mib_max": 11.7109375
}
```

All plots compare lazy / regular replay time. Demotion and boundary time are excluded.
Matching return codes/outcomes do not prove identical output. One trial per mode; regular always ran first.
Missing original return codes are not counted as mismatches.

## Worst 15 calls

### django__django-11728, call 5 (step 2, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 3.485×; 0.4518 → 1.5748 s; +1122.9 ms. Lazy reads: 335.7 MiB.

```bash
grep -r "simplify_regexp" .
```

### django__django-11728, call 4 (step 1, action 4)
Run: qwen-run-20260923T073021Z-LtPzSb. 3.401×; 0.4620 → 1.5714 s; +1109.4 ms. Lazy reads: 335.0 MiB.

```bash
grep -r "replace_named_groups" .
```

### django__django-13023, call 28 (step 26, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 3.280×; 0.3299 → 1.0819 s; +752.1 ms. Lazy reads: 118.3 MiB.

```bash
grep -r "def sanitize_separators" django/
```

### django__django-11728, call 7 (step 3, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 3.058×; 0.3323 → 1.0160 s; +683.7 ms. Lazy reads: 108.9 MiB.

```bash
grep -r "simplify_regexp" --include="*.py" .
```

### django__django-13121, call 30 (step 29, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 3.050×; 0.3469 → 1.0581 s; +711.2 ms. Lazy reads: 120.0 MiB.

```bash
grep -r "Decimal.*duration" django/
```

### django__django-12193, call 2 (step 1, action 2)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.977×; 0.3549 → 1.0565 s; +701.6 ms. Lazy reads: 111.6 MiB.

```bash
find . -type f -name "*.py" | grep -v "__pycache__" | xargs grep -l "CheckboxInput"
```

### django__django-11790, call 3 (step 3, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.901×; 0.3527 → 1.0229 s; +670.3 ms. Lazy reads: 112.0 MiB.

```bash
find . -type f -name "*.py" | grep -v "__pycache__" | xargs grep -l "class UsernameField"
```

### django__django-12325, call 6 (step 6, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.889×; 0.3544 → 1.0240 s; +669.6 ms. Lazy reads: 113.2 MiB.

```bash
find . -type f -name "*.py" | xargs grep -l "Add parent_link=True"
```

### django__django-12858, call 6 (step 6, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.885×; 0.3623 → 1.0452 s; +682.9 ms. Lazy reads: 113.3 MiB.

```bash
find . -type f -name "*.py" | xargs grep -l "get_transform"
```

### django__django-11211, call 4 (step 3, action 1)
Run: qwen-run-20260923T053154Z-PF12hU. 2.863×; 0.3529 → 1.0103 s; +657.3 ms. Lazy reads: 110.1 MiB.

```bash
find . -type f -name "*.py" | xargs grep -l "prefetch_related" | head -20
```

### django__django-12713, call 3 (step 3, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.844×; 0.3682 → 1.0472 s; +679.0 ms. Lazy reads: 111.9 MiB.

```bash
find . -type f -name "*.py" | xargs grep -l "formfield_for_foreignkey"
```

### django__django-11749, call 3 (step 2, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.807×; 0.3758 → 1.0547 s; +679.0 ms. Lazy reads: 112.8 MiB.

```bash
find . -type f -name "*.py" | grep -v "__pycache__" | xargs grep -l "def call_command" | head -10
```

### django__django-11740, call 126 (step 127, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.802×; 0.2584 → 0.7242 s; +465.7 ms. Lazy reads: 140.5 MiB.

```bash
cd /testbed && git reset --hard && sed -i '/self.add_operation(/i \
                # Add dependencies for foreign keys\n                dependencies = []\n                if hasattr(field, "remote_field") and getattr(field.remote_field, "model", None):\n                    dependencies.extend(self._get_dependencies_for_foreign_key(field))' django/db/migrations/autodetector.py && sed -i '/self.add_operation(/ s/$/, dependencies=dependencies)/' django/db/migrations/autodetector.py && grep -A 140 "def generate_altered_fields" django/db/migrations/autodetector.py | grep -A 10 "self.add_operation"
```

### django__django-11740, call 137 (step 138, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.798×; 0.2597 → 0.7265 s; +466.9 ms. Lazy reads: 140.2 MiB.

```bash
cd /testbed && git reset --hard && sed -i '/self.add_operation(/i \
                # Add dependencies for foreign keys\n                dependencies = []\n                if hasattr(field, "remote_field") and getattr(field.remote_field, "model", None):\n                    dependencies.extend(self._get_dependencies_for_foreign_key(field))' django/db/migrations/autodetector.py && sed -i '/self.add_operation(/ s/$/, dependencies=dependencies)/' django/db/migrations/autodetector.py && grep -A 250 "def generate_altered_fields" django/db/migrations/autodetector.py | grep -A 10 "self.add_operation"
```

### django__django-11740, call 147 (step 148, action 1)
Run: qwen-run-20260923T073021Z-LtPzSb. 2.797×; 0.2615 → 0.7315 s; +470.0 ms. Lazy reads: 139.9 MiB.

```bash
cd /testbed && git reset --hard && sed -i '/self.add_operation(/i \
                # Add dependencies for foreign keys\n                dependencies = []\n                if hasattr(field, "remote_field") and getattr(field.remote_field, "model", None):\n                    dependencies.extend(self._get_dependencies_for_foreign_key(field))' django/db/migrations/autodetector.py && sed -i '/self.add_operation(/ s/$/, dependencies=dependencies)/' django/db/migrations/autodetector.py && grep -A 350 "def generate_altered_fields" django/db/migrations/autodetector.py | grep -A 10 "self.add_operation"
```

## Calls differing from the original recording

- django__django-11740, call 49: original rc=0; replay rc=1.
- django__django-12754, call 34: original rc=0; replay rc=1.
- django__django-12754, call 35: original rc=0; replay rc=1.
- django__django-12754, call 36: original rc=0; replay rc=1.
- django__django-12754, call 42: original rc=0; replay rc=1.
