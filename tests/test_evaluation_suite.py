from evals.run_evals import load_cases, run_single_case


def test_all_evaluation_cases_pass():
    cases = load_cases()
    assert len(cases) >= 8
    failures = []
    for case in cases:
        passed, _, details = run_single_case(case)
        if not passed:
            failures.append(f"{case['id']}: {details}")
    assert not failures, "\n".join(failures)
