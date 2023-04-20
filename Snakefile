rule mwej_rates:
    output:
        "src/tex/output/mwej_rates.txt"
    script:
        "src/scripts/mwej_rates.py"