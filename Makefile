.PHONY: install index lint format type-check test api dashboard evaluate docker-up docker-down clean all

install:
	pip install -r requirements.txt -r requirements-dev.txt

index:
	python -c "from src.rag.runtime import PolicyRuntime; r = PolicyRuntime(); print(r.documents_indexed, len(r.sections))"

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/

test:
	pytest tests/ -v --tb=short

api:
	uvicorn src.api.main:app --reload

dashboard:
	streamlit run app.py

evaluate:
	python -m src.evaluation.evaluator

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf data/vectorstore/*

all: install index test
