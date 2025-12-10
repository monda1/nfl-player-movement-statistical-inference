# NFL Big Data Bowl 2026: Análisis Estadístico Avanzado de Movimiento de Jugadores

##  Descripción del Proyecto

Este proyecto presenta un **análisis estadístico inferencial completo** del movimiento de jugadores de la NFL durante jugadas de pase, utilizando datos de seguimiento (tracking data) del NFL Big Data Bowl 2026. El estudio aplica metodologías estadísticas avanzadas para modelar, predecir y comprender el comportamiento espacial-temporal de jugadores mientras el balón está en el aire.

### Objetivo General
Desarrollar modelos estadísticos robustos que permitan:
1. **Predecir trayectorias** de jugadores ofensivos y defensivos post-lanzamiento
2. **Identificar patrones** de cobertura defensiva mediante análisis de distribuciones
3. **Estimar probabilidades** de separación receptor-defensor en función de características del juego
4. **Evaluar el impacto** de variables como velocidad, aceleración y formación en el resultado de la jugada

---

##  Objetivos Específicos del Análisis Estadístico

### 1. Variables Aleatorias Unidimensionales y sus Características

**Variables a Analizar:**
- **Velocidad (s)**: Variable continua medida en yardas/segundo
- **Aceleración (a)**: Variable continua medida en yardas/segundo²
- **Distancia de Separación (d)**: Distancia euclidiana entre receptor objetivo y defensor más cercano
- **Cambio de Posición (Δx, Δy)**: Desplazamiento en el eje longitudinal y transversal

**Análisis a Realizar:**
- Cálculo de **momentos** (media, varianza, asimetría, curtosis)
- Identificación de **valores atípicos** mediante método IQR y Z-scores
- Construcción de **funciones de distribución empírica** (ECDF)
- Análisis de **cuantiles** y percentiles relevantes (P25, P50, P75, P90, P95)

```r
# Ejemplo: Análisis de momentos de velocidad
library(moments)
mean_speed <- mean(data$s)
var_speed <- var(data$s)
skewness_speed <- skewness(data$s)
kurtosis_speed <- kurtosis(data$s)
```

### 2. Distribuciones Usuales en la Práctica Estadística

**Ajuste y Validación de Distribuciones:**

#### Distribuciones Continuas
- **Normal**: Para velocidades promedio de receptores en rutas profundas
- **Exponencial**: Tiempo hasta el primer cambio de dirección del defensor
- **Gamma**: Distancia acumulada recorrida durante la jugada
- **Beta**: Proporción de separación respecto a la separación máxima posible
- **Weibull**: Modelado de tiempos de reacción defensiva

#### Distribuciones Discretas
- **Poisson**: Número de jugadores defensivos en un radio de 5 yardas del receptor
- **Binomial**: Éxito/fracaso en jugadas con características similares
- **Binomial Negativa**: Número de frames hasta cobertura efectiva

**Metodología:**
```r
# Test de bondad de ajuste Kolmogorov-Smirnov
library(fitdistrplus)
fit_normal <- fitdist(data$s, "norm")
fit_gamma <- fitdist(data$s, "gamma")
gofstat(list(fit_normal, fit_gamma))

# QQ-plots para validación visual
qqnorm(data$s)
qqline(data$s, col = "red")
```

### 3. Propiedades de una Muestra Aleatoria

**Validación de Aleatoriedad:**
- **Test de rachas** (Runs Test) para secuencias temporales de frames
- **Prueba de independencia** Chi-cuadrado entre plays consecutivos
- **Análisis de autocorrelación** (ACF/PACF) en series temporales de posición

**Teorema del Límite Central:**
- Demostración empírica con bootstrapping de medias muestrales
- Comparación de distribución muestral con distribución teórica

```r
# Bootstrap para distribución muestral de la media
set.seed(2026)
bootstrap_means <- replicate(10000, {
  sample_data <- sample(data$s, size = 100, replace = TRUE)
  mean(sample_data)
})
hist(bootstrap_means, probability = TRUE)
curve(dnorm(x, mean = mean(data$s), sd = sd(data$s)/sqrt(100)), add = TRUE, col = "red", lwd = 2)
```

### 4. Principios de Reducción de Datos

**Técnicas Aplicadas:**

#### Estadísticos Suficientes
- Identificación de estadísticos suficientes para familias exponenciales
- Factorización de la función de verosimilitud

#### Reducción Dimensional
- **PCA (Análisis de Componentes Principales)**: Reducir 10+ variables de tracking a componentes principales
- **t-SNE**: Visualización de clusters de tipos de jugadas en espacio reducido
- **UMAP**: Alternativa moderna a t-SNE para grandes volúmenes de datos

```r
# PCA para variables de movimiento
library(FactoMineR)
pca_result <- PCA(data[, c("x", "y", "s", "a", "o", "dir")], graph = FALSE)
fviz_pca_biplot(pca_result, repel = TRUE)
```

#### Estadísticos de Orden
- Uso de medianas y percentiles para robustez ante outliers
- Rango intercuartílico para análisis de dispersión

### 5. Métodos para Encontrar Estimadores

#### Método de Máxima Verosimilitud (MLE)
```r
# MLE para distribución de velocidades
library(MASS)
fit_mle <- fitdistr(data$s, "normal")
print(fit_mle)
```

#### Método de Momentos (MoM)
- Estimación de parámetros igualando momentos muestrales con poblacionales
- Aplicación a distribución Gamma y Beta

#### Estimadores Bayesianos
```r
# Prior conjugado para media de velocidad
library(bayesrules)
# Prior: Normal(mu_0 = 5, tau^2 = 2)
# Likelihood: Normal(mu, sigma^2)
posterior_mean <- (sigma^2 * mu_0 + n * tau^2 * xbar) / (sigma^2 + n * tau^2)
```

#### Estimación M-robusta
- Estimadores robustos ante outliers usando función Huber
```r
library(MASS)
rlm(y ~ x, data = data, method = "M")
```

### 6. Introducción a la Inferencia e Imputación

#### Inferencia Estadística
**Tests Paramétricos:**
- Test t para diferencias en velocidad entre posiciones
- ANOVA para comparar múltiples formaciones ofensivas
- Test Z para proporciones de jugadas completadas

**Tests No Paramétricos:**
- Mann-Whitney U para comparaciones sin supuestos de normalidad
- Kruskal-Wallis para múltiples grupos
- Test de Friedman para medidas repetidas

#### Imputación de Datos Faltantes

**Métodos Simples:**
- Imputación por media/mediana condicional
- Last Observation Carried Forward (LOCF)

**Métodos Avanzados:**
```r
# MICE (Multiple Imputation by Chained Equations)
library(mice)
imputed_data <- mice(data, m = 5, method = "pmm", seed = 2026)
completed_data <- complete(imputed_data, 1)

# missForest para datos mixtos
library(missForest)
imputed_rf <- missForest(data, maxiter = 10, ntree = 100)
```

### 7. Métodos para Evaluar Estimadores

#### Propiedades Teóricas
- **Insesgamiento**: E[θ̂] = θ
- **Consistencia**: θ̂ →^P θ cuando n → ∞
- **Eficiencia**: Var(θ̂) alcanza la cota de Cramér-Rao
- **Suficiencia**: T(X) contiene toda la información de θ

#### Evaluación Empírica
```r
# Simulación Monte Carlo para evaluar estimadores
simulate_estimator_performance <- function(n_sims = 1000, sample_size = 100) {
  true_param <- 5.5
  estimates <- replicate(n_sims, {
    sample_data <- rnorm(sample_size, mean = true_param, sd = 2)
    mean(sample_data)  # Estimador
  })
  
  bias <- mean(estimates) - true_param
  mse <- mean((estimates - true_param)^2)
  variance <- var(estimates)
  
  return(list(bias = bias, variance = variance, mse = mse))
}
```

#### Error Cuadrático Medio (MSE)
- MSE = Sesgo² + Varianza
- Comparación entre estimadores alternativos

### 8. Modelos Lineales Generalizados (GLM)

#### Regresión Logística
**Modelo**: Probabilidad de pase completo dado características de la jugada
```r
# Variables predictoras: velocidad, separación, formación, tipo de cobertura
glm_complete <- glm(pass_complete ~ s + separation + offense_formation + 
                    team_coverage_type + defenders_in_box,
                    data = data, family = binomial(link = "logit"))
summary(glm_complete)

# Odds ratios
exp(coef(glm_complete))

# Curva ROC
library(pROC)
roc_curve <- roc(data$pass_complete, predict(glm_complete, type = "response"))
plot(roc_curve, main = paste("AUC =", round(auc(roc_curve), 3)))
```

#### Regresión de Poisson
**Modelo**: Número de defensores en zona de cobertura
```r
glm_poisson <- glm(defenders_count ~ down + yards_to_go + receiver_alignment,
                   data = data, family = poisson(link = "log"))
```

#### Regresión Gamma
**Modelo**: Distancia recorrida por el receptor
```r
glm_gamma <- glm(distance_traveled ~ route_type + player_position + s_initial,
                 data = data, family = Gamma(link = "log"))
```

#### Quasi-Poisson para Sobredispersión
```r
glm_quasipoisson <- glm(defenders_count ~ ., 
                        data = data, family = quasipoisson)
```

### 9. Estimación por Intervalos

#### Intervalos de Confianza Paramétricos
```r
# IC para la media de velocidad (95%)
t.test(data$s, conf.level = 0.95)$conf.int

# IC para diferencia de medias
t.test(s ~ player_side, data = data)

# IC para proporciones
prop.test(x = sum(data$pass_complete), n = nrow(data))
```

#### Intervalos Bootstrap
```r
# Bootstrap percentil
library(boot)
boot_mean <- function(data, indices) {
  return(mean(data[indices]))
}
boot_results <- boot(data$s, boot_mean, R = 10000)
boot.ci(boot_results, type = c("perc", "bca"))
```

#### Intervalos de Predicción
```r
# Para nuevas observaciones en modelo lineal
new_data <- data.frame(s = 6.5, separation = 3.2, defenders_in_box = 6)
predict(lm_model, newdata = new_data, interval = "prediction", level = 0.95)
```

### 10. Tests de Hipótesis

#### Tests Paramétricos

**Test t para una muestra:**
```r
# H0: μ_velocity = 5.0 vs H1: μ_velocity ≠ 5.0
t.test(data$s, mu = 5.0, alternative = "two.sided")
```

**Test t para dos muestras:**
```r
# Comparar velocidad entre offense y defense
t.test(s ~ player_side, data = data, var.equal = FALSE)
```

**ANOVA:**
```r
# Comparar velocidad entre múltiples posiciones
aov_model <- aov(s ~ player_position, data = data)
summary(aov_model)

# Post-hoc con corrección Bonferroni
pairwise.t.test(data$s, data$player_position, p.adjust.method = "bonferroni")
```

**Test Chi-cuadrado:**
```r
# Independencia entre formación y resultado del pase
chisq.test(table(data$offense_formation, data$pass_result))
```

#### Tests No Paramétricos

**Mann-Whitney U:**
```r
wilcox.test(s ~ player_side, data = data)
```

**Kruskal-Wallis:**
```r
kruskal.test(s ~ player_position, data = data)
```

#### Control de Tasa de Error
```r
# Corrección FDR (False Discovery Rate)
p_values <- c(0.001, 0.023, 0.045, 0.089, 0.12)
p.adjust(p_values, method = "fdr")
```

---

## 📈 Análisis Avanzado con Power BI

### Configuración y Preparación de Datos

#### 1. Conexión y Transformación (Power Query)
```m
// Cargar datos desde CSV
let
    Source = Csv.Document(File.Contents("input_2023_w01.csv")),
    Promoted = Table.PromoteHeaders(Source),
    Changed_Type = Table.TransformColumnTypes(Promoted, {
        {"x", type number}, {"y", type number}, {"s", type number},
        {"a", type number}, {"frame_id", Int64.Type}
    })
in
    Changed_Type

// Crear columna calculada: Distancia euclidiana
= SQRT(POWER([x] - [ball_land_x], 2) + POWER([y] - [ball_land_y], 2))

// Normalizar velocidades por posición
= ([s] - [mean_s_position]) / [std_s_position]
```

#### 2. Modelado de Datos
- **Esquema Estrella**: Tabla de hechos (tracking) + dimensiones (players, games, plays)
- **Relaciones**: game_id y play_id como claves foráneas
- **Jerarquías**: Season → Week → Game → Play → Frame

### Visualizaciones Avanzadas

#### 3. DAX Measures Avanzadas

```dax
// Velocidad promedio ponderada por frame
Avg_Weighted_Speed = 
SUMX(
    Tracking,
    [s] * [frame_weight]
) / SUM(Tracking[frame_weight])

// Percentil 90 de separación
P90_Separation = 
PERCENTILE.INC(Tracking[separation_distance], 0.90)

// Tasa de éxito condicional
Success_Rate_Conditional = 
CALCULATE(
    DIVIDE(
        COUNTROWS(FILTER(Plays, [pass_result] = "C")),
        COUNTROWS(Plays)
    ),
    ALLEXCEPT(Plays, Plays[offense_formation], Plays[team_coverage_type])
)

// Moving Average (3 frames)
MA_Speed_3 = 
AVERAGEX(
    FILTER(
        ALL(Tracking),
        Tracking[frame_id] >= EARLIER(Tracking[frame_id]) - 1 &&
        Tracking[frame_id] <= EARLIER(Tracking[frame_id]) + 1 &&
        Tracking[nfl_id] = EARLIER(Tracking[nfl_id])
    ),
    [s]
)

// Tasa de aceleración relativa
Relative_Acceleration = 
VAR CurrentAcc = [a]
VAR PositionAvg = CALCULATE(AVERAGE(Tracking[a]), ALLEXCEPT(Tracking, Tracking[player_position]))
RETURN DIVIDE(CurrentAcc - PositionAvg, PositionAvg)

// Expected Points Added (integración con datos externos)
EPA_Impact = 
SUMX(
    Plays,
    [expected_points_added] * [participation_weight]
)
```

#### 4. Visualizaciones Personalizadas con R/Python

**Heatmap de Densidad de Posiciones:**
```r
# Script R en Power BI
library(ggplot2)
library(viridis)

ggplot(dataset, aes(x = x, y = y)) +
  stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", alpha = 0.7) +
  scale_fill_viridis(option = "plasma") +
  coord_fixed(ratio = 120/53.3) +
  theme_minimal() +
  labs(title = "Densidad de Posiciones - Receptores vs Defensores",
       x = "Yardas (longitudinal)", y = "Yardas (transversal)")
```

**Trayectorias Animadas:**
```python
# Script Python en Power BI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12, 6))
def update(frame):
    ax.clear()
    frame_data = dataset[dataset['frame_id'] == frame]
    ax.scatter(frame_data['x'], frame_data['y'], 
               c=frame_data['player_side'], s=100, alpha=0.7)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    return ax,

anim = FuncAnimation(fig, update, frames=range(1, max_frame), interval=100)
plt.show()
```

#### 5. Dashboards Interactivos Avanzados

**Dashboard 1: Análisis Exploratorio**
- Distribuciones de velocidad/aceleración por posición (violin plots)
- Matriz de correlación entre variables continuas
- Box plots comparativos con detección de outliers
- Tabla dinámica con drill-down por Season → Week → Game

**Dashboard 2: Análisis Predictivo**
- Scatter plot: Separación vs Probabilidad de Completación
- Curvas ROC de modelos GLM importados desde R
- Intervalos de confianza bootstrapeados
- Forecasting de posiciones usando Prophet/ARIMA

**Dashboard 3: Análisis Espacial**
- Campo de fútbol interactivo con trayectorias
- Heatmaps de densidad por zona del campo
- Diagramas de Voronoi para zonas de cobertura
- Animación frame-by-frame con Play As Axis

**Dashboard 4: Insights Estratégicos**
- KPIs: EPA, Success Rate, Separation Rate
- Comparativa formación ofensiva vs tipo de cobertura
- Análisis de tendencias temporales (por semana)
- What-if parameter para simulaciones

#### 6. Técnicas Avanzadas de Power BI

**Field Parameters para Análisis Dinámico:**
```dax
Field_Parameter = {
    ("Velocidad", NAMEOF('Tracking'[s]), 0),
    ("Aceleración", NAMEOF('Tracking'[a]), 1),
    ("Separación", NAMEOF('Tracking'[separation_distance]), 2)
}
```

**Bookmarks y Drillthrough:**
- Bookmarks para cambiar entre vistas de análisis
- Drillthrough pages para análisis detallado de jugadas específicas
- Tooltips personalizados con mini-dashboards

**Integración con R/Python Scripts:**
```r
# Clustering K-means en Power BI
library(cluster)
kmeans_result <- kmeans(dataset[, c("s", "a", "separation")], centers = 4)
dataset$cluster <- as.factor(kmeans_result$cluster)
```

**Publicación y Colaboración:**
- Power BI Service con actualización automática
- Row-Level Security (RLS) por equipo
- Alertas basadas en umbrales de métricas
- Embedded analytics en aplicaciones web

---

##  Stack Tecnológico

### R/RStudio
- **Paquetes estadísticos**: `stats`, `MASS`, `moments`, `fitdistrplus`
- **Inferencia**: `infer`, `broom`, `emmeans`
- **GLM**: `glmnet`, `mgcv`, `gam`
- **Visualización**: `ggplot2`, `plotly`, `gganimate`
- **Machine Learning**: `caret`, `tidymodels`, `mlr3`
- **Imputación**: `mice`, `missForest`, `Amelia`
- **Series temporales**: `forecast`, `tseries`

### Power BI
- **Power Query** (M language)
- **DAX** (Data Analysis Expressions)
- **R/Python visual integration**
- **Custom visuals**: Deneb, Charticulator
- **Power BI Service** para colaboración

### Herramientas Complementarias
- **Git/GitHub** para control de versiones
- **Docker** para reproducibilidad
- **Jupyter Notebooks** para documentación interactiva

---

##  Estructura del Proyecto

```
nfl-player-movement-statistical-analysis/
│
├── data/
│   ├── raw/                          # Datos originales CSV
│   ├── processed/                    # Datos limpios y transformados
│   └── external/                     # Datos externos (nflverse, PFR)
│
├── notebooks/
│   ├── 01_exploratory_analysis.Rmd   # Análisis exploratorio
│   ├── 02_distributions.Rmd          # Ajuste de distribuciones
│   ├── 03_inference.Rmd              # Inferencia estadística
│   ├── 04_glm_models.Rmd             # Modelos GLM
│   ├── 05_hypothesis_testing.Rmd     # Tests de hipótesis
│   └── 06_advanced_analysis.Rmd      # Análisis avanzado
│
├── scripts/
│   ├── data_preprocessing.R          # Limpieza de datos
│   ├── feature_engineering.R         # Creación de variables
│   ├── statistical_tests.R           # Batería de tests
│   ├── glm_modeling.R                # Modelado GLM
│   └── visualization_functions.R     # Funciones de gráficos
│
├── powerbi/
│   ├── NFL_Analysis.pbix             # Archivo Power BI principal
│   ├── data_model.json               # Modelo de datos exportado
│   └── dax_measures.txt              # Documentación de DAX
│
├── reports/
│   ├── statistical_report.pdf        # Reporte técnico completo
│   ├── executive_summary.pdf         # Resumen ejecutivo
│   └── visualizations/               # Gráficos en alta resolución
│
├── tests/
│   └── test_functions.R              # Unit tests para funciones
│
├── docs/
│   ├── methodology.md                # Metodología detallada
│   ├── data_dictionary.md            # Diccionario de variables
│   └── references.bib                # Referencias bibliográficas
│
├── .gitignore
├── README.md
├── requirements.txt                   # Paquetes de Python
├── renv.lock                         # Paquetes de R (renv)
└── LICENSE
```

---

## 🚀 Instrucciones de Uso

### Requisitos Previos
```r
# Instalar paquetes necesarios
install.packages(c(
  "tidyverse", "data.table", "ggplot2", "plotly",
  "MASS", "fitdistrplus", "moments", "car",
  "glmnet", "caret", "mice", "missForest",
  "boot", "infer", "broom", "emmeans",
  "pROC", "FactoMineR", "cluster"
))
```

### Ejecución del Análisis
```bash
# Clonar repositorio
git clone https://github.com/tuusuario/nfl-player-movement-statistical-analysis.git
cd nfl-player-movement-statistical-analysis

# Descargar datos
Rscript scripts/download_data.R

# Ejecutar pipeline completo
Rscript scripts/run_analysis.R
```

### Power BI
1. Abrir `powerbi/NFL_Analysis.pbix`
2. Actualizar conexión de datos a carpeta local `data/processed/`
3. Refrescar datasets
4. Explorar dashboards interactivos

---

##  Resultados Esperados

### Entregables Estadísticos
1. **Reporte de distribuciones**: Ajuste de 10+ distribuciones a variables clave
2. **Matriz de tests**: 50+ tests de hipótesis documentados
3. **Modelos GLM**: Mínimo 5 modelos con validación cruzada
4. **Intervalos de confianza**: Bootstrap y paramétricos para todos los estimadores
5. **Análisis de imputación**: Comparación de métodos MICE, missForest, KNN

### Entregables Visuales (Power BI)
1. **4 Dashboards interactivos** completos
2. **20+ visualizaciones personalizadas** con R/Python
3. **Medidas DAX avanzadas** (>30 measures)
4. **Animaciones** de trayectorias de jugadores

---
