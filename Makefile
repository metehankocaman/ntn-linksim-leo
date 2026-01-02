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
