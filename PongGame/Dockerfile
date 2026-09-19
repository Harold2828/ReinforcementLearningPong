FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_BACKEND_URL=http://localhost:5001
ARG VITE_EVOLUTION_COURT_COUNT=6
ARG VITE_EVOLUTION_POPULATION_SIZE=12
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}
ENV VITE_EVOLUTION_COURT_COUNT=${VITE_EVOLUTION_COURT_COUNT}
ENV VITE_EVOLUTION_POPULATION_SIZE=${VITE_EVOLUTION_POPULATION_SIZE}

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
