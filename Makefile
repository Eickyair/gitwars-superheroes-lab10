# =============================
# Makefile - Automatización Docker
# Proyecto: API Bomba en el IIMAS
# =============================

.PHONY: build run logs status stop clean package clean-package test

IMAGE_NAME=bomba-iimas-api
CONTAINER_NAME=bomba-iimas-container
PORT=8000
TEAM_NAME=bomba_en_el_iimas

# -----------------------------
# Construir la imagen Docker
# -----------------------------
build:
	docker build -t $(IMAGE_NAME) .

# -----------------------------
# Ejecutar contenedor en modo detached
# -----------------------------
run:
	docker run -d -p $(PORT):8000 --name $(CONTAINER_NAME) $(IMAGE_NAME)

# -----------------------------
# Ver logs del contenedor
# -----------------------------
logs:
	docker logs -f $(CONTAINER_NAME)

# -----------------------------
# Ver estado del contenedor
# -----------------------------
status:
	docker ps | grep $(CONTAINER_NAME) || echo "❌ Container not running"

# -----------------------------
# Detener y eliminar el contenedor
# -----------------------------
stop:
	- docker stop $(CONTAINER_NAME)
	- docker rm $(CONTAINER_NAME)

# -----------------------------
# Limpiar recursos de Docker
# -----------------------------
clean:
	docker system prune -f

# -----------------------------
# Borrar paquete viejo (.tar.gz)
# -----------------------------
clean-package:
	rm -f $(TEAM_NAME).tar.gz

# -----------------------------
# Empaquetar TODO el proyecto
# -----------------------------
package: clean-package
	tar --exclude='.git' \
		--exclude='__pycache__' \
		--exclude='*.tar.gz' \
		-czvf ../$(TEAM_NAME).tar.gz .


test:
	python ./src/tests/api_tests.py