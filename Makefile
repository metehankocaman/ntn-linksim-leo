setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && black .

run:
	. .venv/bin/activate && ntnls simulate --snr-db 0 5 10 --seed 1 --out results/

run-cfo:
	. .venv/bin/activate && ntnls cfo-sweep --cfo-hz 0 15000 30000 45000 60000 --snr-db 20 --seed 1 --out results_cfo/

run-delay:
	. .venv/bin/activate && ntnls delay-sweep --delay-samples 0 4 8 12 16 20 24 --snr-db 20 --seed 1 --out results_delay/
