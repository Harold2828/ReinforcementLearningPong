FROM node:20-alpine AS build

WORKDIR /app

ARG VITE_BACKEND_URL=http://localhost:5001
ENV VITE_BACKEND_URL=${VITE_BACKEND_URL}

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
