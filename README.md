# Modelos MA (Moving Average) - Series de Tiempo

Aplicación profesional para análisis y predicción de series temporales usando modelos MA (Moving Average) con datos reales.

## Características

### 📊 Análisis Completo
- Implementación de modelos MA(q) de cualquier orden
- Análisis de estacionariedad (ACF y PACF)
- Diagnóstico completo de residuales
- Tests estadísticos (Ljung-Box, Jarque-Bera)

### 📈 Datos Reales
- **Datos Financieros**: Integración con Yahoo Finance para obtener precios de acciones en tiempo real
- **Datos Económicos**: Datasets de ventas, temperatura y producción industrial

### 🎯 Visualizaciones Profesionales
- Serie temporal original
- Funciones de autocorrelación (ACF y PACF)
- Diagnóstico de residuales (gráfico temporal, histograma, Q-Q plot)
- Predicciones con intervalos de confianza
- Todas las gráficas son interactivas y de alta calidad

### 🔧 Funcionalidades
- Ajuste automático de modelos MA de orden 1 a 10
- Predicciones configurables (5 a 50 períodos)
- Intervalos de confianza al 90%, 95% y 99%
- Métricas de rendimiento (AIC, BIC, MAE, RMSE, MAPE)
- Interpretación automática de resultados

## Instalación

1. Clonar el repositorio:
```bash
git clone <url-repositorio>
cd MODELOS_MA
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecutar la aplicación Streamlit:
```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Interfaz de Usuario

### Panel de Configuración (Sidebar)
1. **Fuente de Datos**: Selecciona entre datos financieros o económicos
2. **Parámetros del Modelo**: Configura el orden MA (q)
3. **Predicción**: Define el número de períodos a predecir
4. **Nivel de Confianza**: Selecciona el intervalo de confianza

### Secciones Principales
1. **Datos de la Serie Temporal**: Estadísticas descriptivas y visualización
2. **Análisis de Estacionariedad**: ACF y PACF
3. **Modelo MA(q)**: Coeficientes y métricas del modelo
4. **Diagnóstico de Residuales**: Análisis completo de residuales
5. **Predicciones**: Forecast con intervalos de confianza
6. **Métricas de Rendimiento**: MAE, RMSE, MAPE
7. **Interpretación**: Explicación detallada del modelo

## Teoría: Modelos MA

Un modelo MA(q) representa una serie temporal como:

Y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_qε_{t-q}

Donde:
- Y_t: valor en el tiempo t
- μ: media de la serie
- ε_t: error en el tiempo t
- θ_i: coeficientes MA
- q: orden del modelo

### Características de los Modelos MA
- La ACF se corta después del lag q
- La PACF decae gradualmente
- Útiles para series con shocks de corta duración
- No requieren diferenciación si la serie es estacionaria

## Tecnologías Utilizadas

- **Streamlit**: Framework para la interfaz web
- **Statsmodels**: Modelado estadístico y series temporales
- **Pandas**: Manipulación de datos
- **NumPy**: Cálculos numéricos
- **Matplotlib/Seaborn**: Visualizaciones
- **Scikit-learn**: Métricas de evaluación
- **yFinance**: Datos financieros en tiempo real
- **SciPy**: Tests estadísticos

## Ejemplos de Uso

### Análisis de Acciones
1. Selecciona "Datos Financieros"
2. Ingresa el símbolo (ej: AAPL, GOOGL, MSFT)
3. Selecciona el período (1 año, 2 años, etc.)
4. Configura el orden MA
5. Analiza los resultados y predicciones

### Análisis de Datos Económicos
1. Selecciona "Datos Económicos"
2. Elige un dataset (Ventas, Temperatura, Producción)
3. Ajusta los parámetros del modelo
4. Visualiza las predicciones

## Interpretación de Resultados

### AIC y BIC
- Valores más bajos indican mejor ajuste
- Útiles para comparar diferentes órdenes MA

### Diagnóstico de Residuales
- Los residuales deben ser ruido blanco
- Test de Ljung-Box: p-valor > 0.05 indica independencia
- Test de Jarque-Bera: p-valor > 0.05 indica normalidad

### Métricas de Error
- **MAE**: Error absoluto promedio
- **RMSE**: Penaliza errores grandes
- **MAPE**: Error porcentual, útil para comparación

## Autor

Desarrollado para análisis profesional de series temporales

## Licencia

MIT License
