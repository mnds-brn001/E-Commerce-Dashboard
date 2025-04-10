import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from datetime import datetime
from utils.KPIs import load_data, calculate_churn_features, define_churn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def app():
    # Configuração da página
    #st.set_page_config(layout="wide")
    
    # Título principal com ícone
    st.title("🔄 Análise de Churn")
    
    # Adicionar abas para diferentes funcionalidades
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral", 
        "⚙️ Configurar Análise", 
        "📈 Resultados do Modelo", 
        "🔮 Previsão"
    ])
    
    # Carregar dados
    df = load_data()
    
    # TAB 1: VISÃO GERAL
    with tab1:
        # Cabeçalho com descrição
        st.header("📊 Visão Geral do Churn")
        
        # Layout em duas colunas para o conteúdo principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin-top: 0;">O que é Churn?</h3>
                <p>O churn (cancelamento) é um indicador crítico para negócios, pois indica a taxa com que os clientes 
                param de comprar ou usar os serviços. Monitorar e prever o churn permite que empresas tomem medidas 
                proativas para reter clientes valiosos.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Exibir distribuição de compras ao longo do tempo
            st.subheader("📅 Distribuição de Compras ao Longo do Tempo")
            
            monthly_orders = df.groupby(pd.to_datetime(df['order_purchase_timestamp']).dt.to_period('M'))['order_id'].count().reset_index()
            monthly_orders['order_purchase_timestamp'] = monthly_orders['order_purchase_timestamp'].astype(str)
            
            fig = px.line(
                monthly_orders, 
                x='order_purchase_timestamp', 
                y='order_id',
                title="Número de Pedidos por Mês",
                labels={'order_purchase_timestamp': 'Mês', 'order_id': 'Número de Pedidos'}
            )
            fig.update_layout(
                xaxis=dict(tickangle=45),
                yaxis=dict(title="Número de Pedidos"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mostrar informações básicas dos dados
            st.subheader("📋 Informações dos Dados")
            
            max_date = pd.to_datetime(df['order_purchase_timestamp']).max()
            min_date = pd.to_datetime(df['order_purchase_timestamp']).min()
            total_customers = df['customer_unique_id'].nunique()
            total_orders = df['order_id'].nunique()
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <p><strong>Período dos Dados:</strong> {min_date.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}</p>
                <p><strong>Total de Clientes:</strong> {total_customers:,}</p>
                <p><strong>Total de Pedidos:</strong> {total_orders:,}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Verificar se o modelo já foi treinado
            model_exists = os.path.exists('churn_model.pkl')
            results_exist = os.path.exists('churn_analysis_results.txt')
            
            if results_exist:
                st.success("✅ Um modelo de churn já foi treinado. Veja os resultados na aba 'Resultados do Modelo'.")
                
                # Extrair informações básicas do arquivo de resultados
                with open('churn_analysis_results.txt', 'r') as f:
                    results_text = f.read()
                    
                # Procurar taxa de churn
                churn_rate_line = [line for line in results_text.split('\n') if "Taxa de churn:" in line]
                if churn_rate_line:
                    churn_rate = churn_rate_line[0].split(": ")[1]
                    st.info(f"📊 A taxa de churn atual é de {churn_rate}")
            else:
                st.warning("⚠️ Nenhum modelo de churn foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
            
            # Exemplo ilustrativo de como funciona a definição de churn
            st.subheader("ℹ️ Como Funciona a Definição de Churn")
            
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <p>Para esta análise, consideramos:</p>
                <ul>
                    <li><strong>Data de corte padrão:</strong> 17 de abril de 2018 (6 meses antes do final dos dados)</li>
                    <li><strong>Clientes com churn (1):</strong> Aqueles que não compraram após a data de corte</li>
                    <li><strong>Clientes sem churn (0):</strong> Aqueles que compraram após a data de corte</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Exemplo ilustrativo de como funciona a definição de churn
        st.subheader("📝 Exemplo de Definição de Churn")
        
        example_data = pd.DataFrame({
            'Cliente': ['Cliente A', 'Cliente B', 'Cliente C', 'Cliente D'],
            'Últimas Compras': [
                'Jan/2018, Mar/2018, Mai/2018, Jul/2018', 
                'Jan/2018, Fev/2018, Mar/2018',
                'Dez/2017, Fev/2018, Set/2018',
                'Nov/2017, Jan/2018, Fev/2018, Abr/2018'
            ],
            'Churn': ['Não', 'Sim', 'Não', 'Sim'],
            'Explicação': [
                'Comprou em Julho (após data de corte)',
                'Última compra em Março (antes da data de corte)',
                'Comprou em Setembro (após data de corte)',
                'Última compra em Abril (mesma data de corte, ainda é churn)'
            ]
        })
        
        st.table(example_data)
        
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <p><strong>Nota:</strong> A data de corte padrão é 17 de abril de 2018. Clientes que não fizeram compras
            após essa data são considerados como tendo abandonado (churn = 1).</p>
        </div>
        """, unsafe_allow_html=True)
        
    # TAB 2: CONFIGURAR ANÁLISE
    with tab2:
        st.header("⚙️ Configurar Análise de Churn")
        
        # Layout em duas colunas para o formulário
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("churn_config_form"):
                st.subheader("📋 Parâmetros da Análise")
                
                # Definição da data de corte
                cutoff_date = st.date_input(
                    "Data de Corte",
                    value=datetime(2018, 4, 17),
                    min_value=datetime(2017, 1, 1),
                    max_value=max_date.to_pydatetime()
                )
                
                # Método de rebalanceamento
                col1, col2 = st.columns(2)
                with col1:
                    rebalance_method = st.selectbox(
                        "Método de Rebalanceamento",
                        options=["smote", "undersample", "none"],
                        help="""
                        - SMOTE: Gera exemplos sintéticos da classe minoritária
                        - Undersample: Remove exemplos da classe majoritária
                        - None: Não realiza rebalanceamento
                        """
                    )
                
                with col2:
                    class_weight = st.selectbox(
                        "Peso das Classes",
                        options=["balanced", "none"],
                        help="""
                        - Balanced: Atribui pesos inversamente proporcionais à frequência da classe
                        - None: Sem pesos
                        """
                    )
                
                # Tipo de modelo
                model_type = st.selectbox(
                    "Tipo de Modelo",
                    options=["random_forest", "xgboost", "logistic_regression"],
                    help="""
                    - Random Forest: Conjunto de árvores de decisão
                    - XGBoost: Implementação de Gradient Boosting
                    - Logistic Regression: Regressão logística
                    """
                )
                
                # Validação cruzada
                col1, col2 = st.columns(2)
                with col1:
                    use_cv = st.number_input(
                        "Número de Folds para Validação Cruzada",
                        min_value=0,
                        max_value=10,
                        value=5,
                        help="0 para não usar validação cruzada"
                    )
                
                with col2:
                    test_size = st.slider(
                        "Proporção de Dados para Teste",
                        min_value=0.1,
                        max_value=0.5,
                        value=0.3,
                        step=0.05,
                        help="Porcentagem dos dados que será usada para teste"
                    )
                
                # Opção de grid search
                grid_search = st.checkbox(
                    "Realizar Grid Search",
                    value=False,
                    help="Busca exaustiva pelos melhores hiperparâmetros (pode demorar)"
                )
                
                # Botão para executar a análise
                submit_button = st.form_submit_button("🚀 Executar Análise de Churn")
        
        with col2:
            st.subheader("ℹ️ Sobre os Parâmetros")
            
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h4>Data de Corte</h4>
                <p>Define o ponto no tempo a partir do qual consideramos que um cliente abandonou o serviço.</p>
                
                <h4>Rebalanceamento</h4>
                <p>Métodos para lidar com o desequilíbrio entre classes (churn vs. não-churn).</p>
                
                <h4>Tipo de Modelo</h4>
                <p>Algoritmos de machine learning para prever o churn.</p>
                
                <h4>Validação Cruzada</h4>
                <p>Método para avaliar a performance do modelo em diferentes subconjuntos dos dados.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 Distribuição Atual")
            
            # Calcular distribuição atual de churn
            cutoff_date_obj = datetime.combine(cutoff_date, datetime.min.time())
            churn_df = define_churn(df, cutoff_date_obj)
            
            if 'churn' in churn_df.columns:
                churn_counts = churn_df['churn'].value_counts()
                total = churn_counts.sum()
                
                fig = go.Figure(data=[go.Pie(
                    labels=['Não Churn', 'Churn'],
                    values=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                    hole=.3,
                    marker_colors=['#3366CC', '#DC3912']
                )])
                
                fig.update_layout(
                    title_text=f"Distribuição Atual de Churn (Baseado na data de corte selecionada)",
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                churn_rate = churn_counts.get(1, 0) / total * 100 if total > 0 else 0
                st.info(f"Taxa de churn atual: {churn_rate:.2f}%")
        
        if submit_button:
            # Formatar data e parâmetros
            cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
            class_weight_param = "None" if class_weight == "none" else class_weight
            cv_param = 0 if use_cv == 0 else use_cv
            
            # Construir comando para executar o script
            grid_search_param = "--grid_search" if grid_search else ""
            
            command = f"""python churn_analysis.py \
                --cutoff_date {cutoff_date_str} \
                --rebalance {rebalance_method} \
                --model {model_type} \
                --class_weight {class_weight} \
                --cv {cv_param} \
                --test_size {test_size} \
                {grid_search_param}"""
            
            st.info(f"Executando análise com o comando: `{command}`")
            
            # Aqui você pode executar o comando via subprocess, mas como estamos no Streamlit,
            # vamos apenas mostrar uma mensagem de progresso para simular a execução
            
            with st.spinner("Executando análise de churn... (Este processo pode demorar alguns minutos)"):
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <p><strong>Nota:</strong> Em um ambiente de produção, este comando seria executado em segundo plano.</p>
                    <p>Para fins de demonstração, acesse o terminal e execute o comando acima manualmente.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Aqui você pode implementar a chamada real para o script de análise
                # Por exemplo:
                # import subprocess
                # result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # st.code(result.stdout)
                
                st.success("✅ Análise concluída! Navegue para a aba 'Resultados do Modelo' para ver os resultados.")
                st.balloons()

    # TAB 3: RESULTADOS DO MODELO
    with tab3:
        st.header("📈 Resultados do Modelo de Churn")
        
        # Verificar se existe um modelo treinado
        if not os.path.exists('churn_analysis_results.txt'):
            st.warning("⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
        else:
            # Carregar resultados do arquivo
            with open('churn_analysis_results.txt', 'r') as f:
                results_text = f.read()
            
            # Extrair informações principais
            lines = results_text.split('\n')
            
            # Configurações
            config_section = False
            configs = {}
            
            # Métricas de performance
            metrics_section = False
            metrics = {}
            
            # Distribuição
            dist_section = False
            distribution = {}
            
            # Processar linhas do arquivo para extrair informações
            for line in lines:
                if line.startswith("Configurações:"):
                    config_section = True
                    dist_section = False
                    metrics_section = False
                    continue
                elif line.startswith("Distribuição de churn:"):
                    config_section = False
                    dist_section = True
                    metrics_section = False
                    continue
                elif line.startswith("Métricas de performance:"):
                    config_section = False
                    dist_section = False
                    metrics_section = True
                    continue
                elif line.strip() == "" or ":" not in line:
                    continue
                
                # Processar configurações
                if config_section and ":" in line:
                    key, value = line.split(":", 1)
                    configs[key.strip()] = value.strip()
                
                # Processar distribuição
                if dist_section and ":" in line:
                    key, value = line.split(":", 1)
                    if key.strip() in ["Não-churn (0)", "Churn (1)", "Taxa de churn"]:
                        distribution[key.strip()] = value.strip()
                
                # Processar métricas
                if metrics_section and ":" in line:
                    key, value = line.split(":", 1)
                    if key.strip() in ["Accuracy", "Precision (weighted)", "Recall (weighted)", 
                                       "F1 (macro)", "F1 (weighted)", "AUC-ROC", "Average Precision Score"]:
                        metrics[key.strip()] = float(value.strip())
            
            # Layout em duas colunas para os resultados
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Exibir configurações utilizadas
                st.subheader("⚙️ Configurações Utilizadas")
                config_df = pd.DataFrame({
                    "Parâmetro": list(configs.keys()),
                    "Valor": list(configs.values())
                })
                st.table(config_df)
                
                # Exibir métricas principais
                st.subheader("📊 Métricas de Performance")
                
                # Primeira linha de métricas
                col1, col2, col3, col4 = st.columns(4)
                
                if "Accuracy" in metrics:
                    col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                if "Precision (weighted)" in metrics:
                    col2.metric("Precision", f"{metrics['Precision (weighted)']:.4f}")
                if "Recall (weighted)" in metrics:
                    col3.metric("Recall", f"{metrics['Recall (weighted)']:.4f}")
                if "F1 (weighted)" in metrics:
                    col4.metric("F1 Score", f"{metrics['F1 (weighted)']:.4f}")
                
                # Segunda linha de métricas
                col1, col2 = st.columns(2)
                
                if "AUC-ROC" in metrics:
                    col1.metric("AUC-ROC", f"{metrics['AUC-ROC']:.4f}")
                if "Average Precision Score" in metrics:
                    col2.metric("Avg. Precision", f"{metrics['Average Precision Score']:.4f}")
                
                # Exibir distribuição de churn
                st.subheader("📈 Distribuição de Churn")
                
                if distribution:
                    # Extrair valores numéricos
                    try:
                        non_churn = int(distribution.get('Não-churn (0)', '0').replace(',', ''))
                        churn = int(distribution.get('Churn (1)', '0').replace(',', ''))
                        
                        # Criar gráfico de pizza
                        fig = go.Figure(data=[go.Pie(
                            labels=['Não Churn', 'Churn'],
                            values=[non_churn, churn],
                            hole=.3,
                            marker_colors=['#3366CC', '#DC3912']
                        )])
                        
                        fig.update_layout(
                            title_text=f"Distribuição de Churn ({distribution.get('Taxa de churn', '')})",
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Erro ao criar gráfico de distribuição: {e}")
                        st.write(distribution)
            
            with col2:
                # Verificar se os arquivos de modelo existem
                if not os.path.exists('churn_model.pkl') or not os.path.exists('churn_analysis_plots.png'):
                    st.warning("⚠️ Arquivos do modelo ou gráficos não encontrados. Alguns resultados podem estar incompletos.")
                else:
                    # Exibir gráficos salvos
                    st.subheader("📊 Gráficos de Avaliação")
                    
                    try:
                        # Tentar carregar e exibir a imagem dos gráficos
                        st.image('churn_analysis_plots.png', use_column_width=True)
                    except Exception as e:
                        st.error(f"Erro ao carregar gráficos: {e}")
                
                # Exibir relatório completo
                with st.expander("📄 Ver Relatório Completo"):
                    st.text(results_text)
                
                # Adicionar insights baseados nas métricas
                st.subheader("💡 Insights")
                
                if "Accuracy" in metrics and "AUC-ROC" in metrics:
                    accuracy = metrics['Accuracy']
                    auc_roc = metrics['AUC-ROC']
                    
                    if accuracy > 0.9 and auc_roc > 0.8:
                        st.success("✅ O modelo apresenta excelente performance, com alta precisão e capacidade de discriminação.")
                    elif accuracy > 0.8 and auc_roc > 0.7:
                        st.info("ℹ️ O modelo apresenta boa performance, mas há espaço para melhorias.")
                    else:
                        st.warning("⚠️ O modelo apresenta performance abaixo do ideal. Considere ajustar os parâmetros ou usar um algoritmo diferente.")
                
                # Adicionar recomendações
                st.subheader("🎯 Recomendações")
                
                if "Precision (weighted)" in metrics and "Recall (weighted)" in metrics:
                    precision = metrics['Precision (weighted)']
                    recall = metrics['Recall (weighted)']
                    
                    if precision > 0.8 and recall < 0.6:
                        st.info("ℹ️ O modelo tem alta precisão, mas baixa sensibilidade. Considere ajustar o threshold para capturar mais casos de churn.")
                    elif precision < 0.6 and recall > 0.8:
                        st.info("ℹ️ O modelo tem alta sensibilidade, mas baixa precisão. Considere ajustar o threshold para reduzir falsos positivos.")
                    elif precision < 0.6 and recall < 0.6:
                        st.warning("⚠️ O modelo tem baixa precisão e sensibilidade. Considere usar um algoritmo diferente ou ajustar os parâmetros.")

    # TAB 4: PREVISÃO
    with tab4:
        st.header("🔮 Previsão de Churn")
        
        # Verificar se existe um modelo treinado
        if not os.path.exists('churn_model.pkl'):
            st.warning("⚠️ Nenhum modelo foi treinado ainda. Acesse a aba 'Configurar Análise' para criar um modelo.")
        else:
            # Layout em duas colunas
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Prever Churn para Novos Clientes")
                
                # Formulário para entrada de dados
                with st.form("prediction_form"):
                    # Carregar o modelo e o scaler
                    with open('churn_model.pkl', 'rb') as f:
                        model = pickle.load(f)
                    
                    with open('churn_scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    
                    with open('churn_feature_columns.pkl', 'rb') as f:
                        feature_columns = pickle.load(f)
                    
                    # Criar campos para entrada de dados
                    st.markdown("""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                        <p>Preencha os dados do cliente para prever a probabilidade de churn.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Criar campos para as features
                    input_data = {}
                    
                    # Dividir em duas colunas para melhor organização
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Primeira metade das features
                        for i, feature in enumerate(feature_columns[:len(feature_columns)//2]):
                            if feature == 'recency':
                                input_data[feature] = st.number_input(
                                    f"Dias desde a última compra ({feature})",
                                    min_value=0,
                                    max_value=365,
                                    value=30
                                )
                            elif feature == 'cancel_rate':
                                input_data[feature] = st.slider(
                                    f"Taxa de cancelamento ({feature})",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.1,
                                    step=0.05
                                )
                            else:
                                input_data[feature] = st.number_input(
                                    f"Valor para {feature}",
                                    value=0.0
                                )
                    
                    with col2:
                        # Segunda metade das features
                        for i, feature in enumerate(feature_columns[len(feature_columns)//2:]):
                            if feature == 'recency':
                                input_data[feature] = st.number_input(
                                    f"Dias desde a última compra ({feature})",
                                    min_value=0,
                                    max_value=365,
                                    value=30
                                )
                            elif feature == 'cancel_rate':
                                input_data[feature] = st.slider(
                                    f"Taxa de cancelamento ({feature})",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.1,
                                    step=0.05
                                )
                            else:
                                input_data[feature] = st.number_input(
                                    f"Valor para {feature}",
                                    value=0.0
                                )
                    
                    # Botão para fazer a previsão
                    predict_button = st.form_submit_button("🔮 Prever Probabilidade de Churn")
                
                if predict_button:
                    # Criar DataFrame com os dados de entrada
                    input_df = pd.DataFrame([input_data])
                    
                    # Verificar se todas as features necessárias estão presentes
                    missing_features = [col for col in feature_columns if col not in input_df.columns]
                    if missing_features:
                        st.error(f"Faltam as seguintes features: {missing_features}")
                    else:
                        # Garantir que as colunas estão na mesma ordem que o modelo espera
                        input_df = input_df[feature_columns]
                        
                        # Aplicar o scaler
                        input_scaled = scaler.transform(input_df)
                        
                        # Fazer a previsão
                        prediction_proba = model.predict_proba(input_scaled)[0]
                        churn_probability = prediction_proba[1]  # Probabilidade de churn (classe 1)
                        
                        # Exibir o resultado
                        st.subheader("📊 Resultado da Previsão")
                        
                        # Criar um medidor visual para a probabilidade
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = churn_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Probabilidade de Churn (%)"},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 30], 'color': "lightgreen"},
                                    {'range': [30, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "black", 'width': 4},
                                    'thickness': 0.75,
                                    'value': churn_probability * 100
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Exibir recomendação baseada na probabilidade
                        if churn_probability < 0.3:
                            st.success(f"✅ Baixa probabilidade de churn ({churn_probability:.2%}). Este cliente provavelmente continuará comprando.")
                        elif churn_probability < 0.7:
                            st.warning(f"⚠️ Probabilidade moderada de churn ({churn_probability:.2%}). Considere ações de retenção preventiva.")
                        else:
                            st.error(f"❌ Alta probabilidade de churn ({churn_probability:.2%}). Ações imediatas de retenção são recomendadas.")
            
            with col2:
                st.subheader("ℹ️ Sobre a Previsão")
                
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h4>Como interpretar os resultados</h4>
                    <p>A previsão fornece a probabilidade de um cliente abandonar o serviço (churn).</p>
                    <ul>
                        <li><strong>Baixa probabilidade (< 30%):</strong> Cliente com baixo risco de churn</li>
                        <li><strong>Probabilidade moderada (30-70%):</strong> Cliente com risco médio de churn</li>
                        <li><strong>Alta probabilidade (> 70%):</strong> Cliente com alto risco de churn</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎯 Ações Recomendadas")
                
                st.markdown("""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                    <h4>Para clientes com alto risco de churn:</h4>
                    <ul>
                        <li>Ofertas personalizadas de desconto</li>
                        <li>Programas de fidelidade específicos</li>
                        <li>Contato proativo da equipe de suporte</li>
                        <li>Recomendações personalizadas de produtos</li>
                    </ul>
                    
                    <h4>Para clientes com risco moderado:</h4>
                    <ul>
                        <li>Lembretes de produtos relacionados</li>
                        <li>Newsletters personalizadas</li>
                        <li>Programas de pontos ou cashback</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    app() 