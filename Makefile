COMPOSE := docker compose
PROJECT_ROOT := $(CURDIR)

.PHONY: production prod clean-cache test test-backend test-frontend build-production up-production reset-backend reset-back reset-frontend reset-front pretrain train-self-play reset-models status logs down

production: clean-cache test build-production up-production status

prod: production

clean-cache:
	@echo "Cleaning local build and test artifacts..."
	rm -rf .pytest_cache Server/.pytest_cache Server/app/__pycache__ Server/app/ai/__pycache__ Server/tests/__pycache__ PongGame/dist PongGame/.vite
	@echo "Stopping existing project containers..."
	$(COMPOSE) down --remove-orphans
	@echo "Pruning Docker builder cache..."
	docker builder prune -f

test: test-backend test-frontend

test-backend:
	python3 -m pytest Server/tests

test-frontend:
	cd PongGame && npm test
	cd PongGame && npm run build
	rm -rf PongGame/dist

build-production:
	$(COMPOSE) build --no-cache

up-production:
	$(COMPOSE) up -d

reset-backend:
	$(COMPOSE) up -d --build --force-recreate backend

reset-back: reset-backend

reset-frontend:
	$(COMPOSE) up -d --build --force-recreate frontend

reset-front: reset-frontend

pretrain:
	$(COMPOSE) run --rm --build backend python pretrain_self_play.py --episodes $${EPISODES:-500} --max-steps $${MAX_STEPS:-1000}

train-self-play: up-production
	@echo "Open http://localhost:5173 and select 'Training Self-Play' in the game mode selector."

reset-models:
	@if [ "$(CONFIRM)" != "reset" ]; then \
		echo "This deletes saved Q-learning model volumes. Re-run with: make reset-models CONFIRM=reset"; \
		exit 1; \
	fi
	$(COMPOSE) down
	docker volume rm reinforcementlearningpong_q_learning_models || true

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f

down:
	$(COMPOSE) down
