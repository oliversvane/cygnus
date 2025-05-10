terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = ">= 2.0.0"
    }
  }
}

provider "kubernetes" {
  host                   = var.k3s_endpoint
  client_certificate     = file("${path.module}/certs/client.crt")
  client_key             = file("${path.module}/certs/client.key")
  cluster_ca_certificate = file("${path.module}/certs/ca.crt")
}

provider "helm" {
  kubernetes {
    host                   = var.k3s_endpoint
    client_certificate     = file("${path.module}/certs/client.crt")
    client_key             = file("${path.module}/certs/client.key")
    cluster_ca_certificate = file("${path.module}/certs/ca.crt")
  }
}

# Namespaces
resource "kubernetes_namespace" "lab_system" {
  metadata {
    name = "lab-system"
  }
}

# TimescaleDB via Bitnami PostgreSQL
resource "helm_release" "timescaledb" {
  name       = "timescaledb"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "postgresql"
  namespace  = kubernetes_namespace.lab_system.metadata[0].name

  set {
    name  = "postgresqlDatabase"
    value = "timescaledb"
  }
  set {
    name  = "postgresqlUsername"
    value = "tsdb_user"
  }
  set {
    name  = "postgresqlPassword"
    value = var.timescaledb_password
  }
  set {
    name  = "primary.persistence.size"
    value = "20Gi"
  }
}

# MinIO
resource "helm_release" "minio" {
  name       = "minio"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "minio"
  namespace  = kubernetes_namespace.lab_system.metadata[0].name

  set {
    name  = "accessKey.password"
    value = var.minio_access_key
  }
  set {
    name  = "secretKey.password"
    value = var.minio_secret_key
  }
  set {
    name  = "persistence.size"
    value = "50Gi"
  }
}

# EMQX
resource "helm_release" "emqx" {
  name       = "emqx"
  repository = "https://repos.emqx.io/charts"
  chart      = "emqx"
  namespace  = kubernetes_namespace.lab_system.metadata[0].name

  set {
    name  = "persistence.size"
    value = "10Gi"
  }
  set {
    name  = "cluster.enabled"
    value = "true"
  }
}

# Middleware Deployment
resource "kubernetes_deployment" "middleware" {
  metadata {
    name      = "middleware"
    namespace = kubernetes_namespace.lab_system.metadata[0].name
  }
  spec {
    replicas = 2
    selector {
      match_labels = { app = "middleware" }
    }
    template {
      metadata {
        labels = { app = "middleware" }
      }
      spec {
        container {
          name  = "api"
          image = var.middleware_image
          port {
            container_port = 8080
          }
          readiness_probe {
            http_get {
              path = "/healthz"
              port = 8080
            }
            initial_delay_seconds = 5
            timeout_seconds        = 3
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "middleware_svc" {
  metadata {
    name      = "middleware-svc"
    namespace = kubernetes_namespace.lab_system.metadata[0].name
  }
  spec {
    selector = { app = "middleware" }
    port {
      port        = 80
      target_port = 8080
    }
    type = "ClusterIP"
  }
}

# Frontend Deployment
resource "kubernetes_deployment" "frontend" {
  metadata {
    name      = "frontend"
    namespace = kubernetes_namespace.lab_system.metadata[0].name
  }
  spec {
    replicas = 1
    selector {
      match_labels = { app = "frontend" }
    }
    template {
      metadata {
        labels = { app = "frontend" }
      }
      spec {
        container {
          name  = "ui"
          image = var.frontend_image
          port {
            container_port = 3000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "frontend_svc" {
  metadata {
    name      = "frontend-svc"
    namespace = kubernetes_namespace.lab_system.metadata[0].name
  }
  spec {
    selector = { app = "frontend" }
    port {
      port        = 80
      target_port = 3000
    }
    type = "ClusterIP"
  }
}

