import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils.KPIs import load_data, calculate_kpis, calculate_acquisition_retention_kpis, filter_by_date_range
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuração da página
st.set_page_config(
    page_title="Dashboard Olist",
    page_icon="📊",
    layout="wide"
)

# Carregar dados para obter o período disponível
df = load_data()
min_date = pd.to_datetime(df['order_purchase_timestamp']).min()
max_date = pd.to_datetime(df['order_purchase_timestamp']).max()

# Sidebar
st.sidebar.title("Configurações")

# Filtro de período
st.sidebar.subheader("Período de Análise")
periodo = st.sidebar.selectbox(
    "Selecione o período:",
    [
        "Todo o período",
        "Último mês",
        "Último trimestre",
        "Último semestre",
        "Último ano",
        "Últimos 2 anos"
    ]
)

# Calcular o período selecionado
def get_date_range(periodo):
    hoje = max_date
    if periodo == "Todo o período":
        return None
    elif periodo == "Último mês":
        return [hoje - timedelta(days=30), hoje]
    elif periodo == "Último trimestre":
        return [hoje - timedelta(days=90), hoje]
    elif periodo == "Último semestre":
        return [hoje - timedelta(days=180), hoje]
    elif periodo == "Último ano":
        return [hoje - timedelta(days=365), hoje]
    elif periodo == "Últimos 2 anos":
        return [hoje - timedelta(days=730), hoje]

# Aplicar filtro de data
date_range = get_date_range(periodo)
filtered_df = filter_by_date_range(df, date_range)

# Filtro de gasto com marketing
st.sidebar.subheader("Total Gasto com Marketing")
marketing_spend = st.sidebar.number_input(
    "Valor (R$):",
    min_value=0,
    max_value=5000000,
    value=50000,
    step=1000,
    help="Digite o valor total gasto com marketing no período selecionado"
)

# Navegação
st.sidebar.markdown("---")
st.sidebar.title("Navegação")
pagina = st.sidebar.radio(
    "Selecione a página:",
    ["Visão Geral", "Análise Estratégica", "Aquisição e Retenção", "Comportamento do Cliente", "Produtos e Categorias"]
)

# Funções auxiliares
def format_value(value, is_integer=False):
    """Formata um valor numérico com separador de milhares e duas casas decimais."""
    if is_integer:
        return f"{int(value):,}"
    return f"{value:,.2f}"

def format_percentage(value):
    """Formata um valor como porcentagem com duas casas decimais."""
    return f"{value*100:.2f}%"

# Exibir a página selecionada
if pagina == "Visão Geral":
    st.title("Visão Geral")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    
    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)
    
    # Primeira linha de KPIs
    col1.metric("💰 Receita Total", f"R$ {format_value(kpis['total_revenue'])}")
    col2.metric("📦 Total de Pedidos", format_value(kpis['total_orders'], is_integer=True))
    col3.metric("👥 Total de Clientes", format_value(kpis['total_customers'], is_integer=True))
    
    # Segunda linha de KPIs
    col1.metric("🎯 Taxa de Abandono", format_percentage(kpis['abandonment_rate']))
    col2.metric("😊 Satisfação do Cliente", format_value(kpis['csat']))
    col3.metric("💰 Ticket Médio", f"R$ {format_value(kpis['average_ticket'])}")
    
    # Terceira linha de KPIs
    col1.metric("📦 Tempo Médio de Entrega", f"{int(kpis['avg_delivery_time'])} dias")
    col2.metric("❌ Taxa de Cancelamento", format_percentage(kpis['cancellation_rate']))
    col3.metric("💸 Receita Perdida", f"R$ {format_value(kpis['lost_revenue'])}")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Receita ao Longo do Tempo
        st.subheader("💰 Receita ao Longo do Tempo")
        monthly_revenue = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['price'].sum().reset_index()
        monthly_revenue['order_purchase_timestamp'] = monthly_revenue['order_purchase_timestamp'].astype(str)
        fig_revenue = px.line(
            monthly_revenue,
            x='order_purchase_timestamp',
            y='price',
            title="Evolução da Receita",
            labels={'price': 'Receita (R$)', 'order_purchase_timestamp': 'Mês'}
        )
        fig_revenue.update_layout(showlegend=False)
        fig_revenue.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_revenue, use_container_width=True)
        
        # Gráfico de Satisfação do Cliente
        st.subheader("😊 Satisfação do Cliente")
        monthly_satisfaction = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['review_score'].mean().reset_index()
        monthly_satisfaction['order_purchase_timestamp'] = monthly_satisfaction['order_purchase_timestamp'].astype(str)
        fig_satisfaction = px.line(
            monthly_satisfaction,
            x='order_purchase_timestamp',
            y='review_score',
            title="Evolução da Satisfação",
            labels={'review_score': 'Nota Média', 'order_purchase_timestamp': 'Mês'}
        )
        fig_satisfaction.update_layout(
            yaxis=dict(range=[0, 5]),
            showlegend=False
        )
        fig_satisfaction.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
    with col2:
        # Gráfico de Taxa de Cancelamento
        st.subheader("❌ Taxa de Cancelamento")
        monthly_cancellation = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['pedido_cancelado'].mean().reset_index()
        monthly_cancellation['order_purchase_timestamp'] = monthly_cancellation['order_purchase_timestamp'].astype(str)
        fig_cancellation = px.line(
            monthly_cancellation,
            x='order_purchase_timestamp',
            y='pedido_cancelado',
            title="Evolução da Taxa de Cancelamento",
            labels={'pedido_cancelado': 'Taxa de Cancelamento', 'order_purchase_timestamp': 'Mês'}
        )
        fig_cancellation.update_layout(
            yaxis=dict(tickformat=".1%"),
            showlegend=False
        )
        fig_cancellation.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_cancellation, use_container_width=True)

elif pagina == "Análise Estratégica":
    st.title("Análise Estratégica")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    
    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)
    
    # Primeira linha de KPIs - Métricas de Receita
    col1.metric("💰 Receita Total", f"R$ {format_value(kpis['total_revenue'])}")
    col2.metric("📈 Ticket Médio", f"R$ {format_value(kpis['average_ticket'])}")
    col3.metric("👥 Total de Clientes", format_value(kpis['total_customers'], is_integer=True))
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Previsão de Receita (curto prazo)
        st.subheader("📊 Previsão de Receita (Próximos 30 dias)")
        
        # Calcular média diária de receita
        filtered_df['date'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.date
        daily_revenue = filtered_df.groupby('date')['price'].sum().reset_index()
        
        # Adicionar dia da semana para análise de sazonalidade
        daily_revenue['day_of_week'] = pd.to_datetime(daily_revenue['date']).dt.day_name()
        
        # Calcular média móvel de 7 dias
        daily_revenue['ma7'] = daily_revenue['price'].rolling(window=7).mean()
        
        # Calcular fatores de sazonalidade semanal
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_seasonality = daily_revenue.groupby('day_of_week')['price'].mean().reindex(day_order)
        weekly_seasonality = weekly_seasonality / weekly_seasonality.mean()  # Normalizar
        
        # Calcular tendência de crescimento (últimos 30 dias)
        recent_data = daily_revenue.tail(30)
        if len(recent_data) >= 2:
            x = np.arange(len(recent_data))
            y = recent_data['price'].values
            z = np.polyfit(x, y, 1)
            growth_rate = z[0]  # Coeficiente de crescimento diário
        else:
            growth_rate = 0
        
        # Calcular previsão para os próximos 30 dias
        last_date = daily_revenue['date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=31, freq='D')[1:]
        
        # Criar DataFrame para previsão
        forecast_df = pd.DataFrame({'date': forecast_dates})
        forecast_df['day_of_week'] = forecast_df['date'].dt.day_name()
        
        # Aplicar fatores de sazonalidade
        forecast_df['seasonality_factor'] = forecast_df['day_of_week'].map(weekly_seasonality)
        
        # Calcular previsão base
        base_forecast = daily_revenue['ma7'].iloc[-1]
        
        # Aplicar tendência de crescimento e sazonalidade
        for i in range(len(forecast_df)):
            days_ahead = i + 1
            forecast_df.loc[i, 'forecast'] = base_forecast * forecast_df.loc[i, 'seasonality_factor'] + (growth_rate * days_ahead)
        
        # Calcular intervalo de confiança (simplificado)
        std_dev = daily_revenue['price'].std()
        forecast_df['lower_bound'] = forecast_df['forecast'] - (1.96 * std_dev)
        forecast_df['upper_bound'] = forecast_df['forecast'] + (1.96 * std_dev)
        
        # Criar gráfico de previsão
        fig_forecast = go.Figure()
        
        # Adicionar dados históricos
        fig_forecast.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['price'],
            name='Receita Real',
            line=dict(color='#1f77b4')
        ))
        
        # Adicionar média móvel
        fig_forecast.add_trace(go.Scatter(
            x=daily_revenue['date'],
            y=daily_revenue['ma7'],
            name='Média Móvel (7 dias)',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Adicionar previsão
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['forecast'],
            name='Previsão (30 dias)',
            line=dict(color='#2ca02c', dash='dot')
        ))
        
        # Adicionar intervalo de confiança
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(44, 160, 44, 0.2)',
            line=dict(color='rgba(44, 160, 44, 0)'),
            name='Intervalo de Confiança (95%)',
            showlegend=True
        ))
        
        fig_forecast.update_layout(
            title="Previsão de Receita para os Próximos 30 Dias",
            xaxis_title="Data",
            yaxis_title="Receita (R$)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig_forecast.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Adicionar métricas de previsão
        st.subheader("📈 Métricas de Previsão")
        col1_metrics, col2_metrics, col3_metrics = st.columns(3)
        
        # Calcular receita total prevista para os próximos 30 dias
        total_forecast = forecast_df['forecast'].sum()
        col1_metrics.metric("💰 Receita Total Prevista (30 dias)", f"R$ {format_value(total_forecast)}")
        
        # Calcular crescimento previsto em relação ao período anterior
        previous_30_days = daily_revenue.tail(30)['price'].sum()
        growth_percentage = (total_forecast - previous_30_days) / previous_30_days * 100 if previous_30_days > 0 else 0
        col2_metrics.metric("📈 Crescimento Previsto", f"{format_value(growth_percentage)}%")
        
        # Calcular dia com maior receita prevista
        max_day = forecast_df.loc[forecast_df['forecast'].idxmax()]
        col3_metrics.metric("📅 Dia com Maior Receita Prevista", f"{max_day['date'].strftime('%d/%m/%Y')} ({max_day['day_of_week']})")
    
    with col2:
        # Sazonalidade de Vendas
        st.subheader("📅 Sazonalidade de Vendas")
        
        # Calcular vendas por dia da semana
        filtered_df['day_of_week'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_revenue = filtered_df.groupby('day_of_week')['price'].sum().reindex(day_order)
        
        # Calcular vendas por mês
        filtered_df['month'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        month_revenue = filtered_df.groupby('month')['price'].sum().reindex(month_order)
        
        # Criar gráfico de sazonalidade
        fig_seasonality = go.Figure()
        
        # Adicionar barras para dia da semana
        fig_seasonality.add_trace(go.Bar(
            x=day_revenue.index,
            y=day_revenue.values,
            name='Por Dia da Semana',
            marker_color='#1f77b4'
        ))
        
        # Adicionar barras para mês
        fig_seasonality.add_trace(go.Bar(
            x=month_revenue.index,
            y=month_revenue.values,
            name='Por Mês',
            marker_color='#ff7f0e',
            visible=False
        ))
        
        # Adicionar botões para alternar entre visualizações
        fig_seasonality.update_layout(
            title="Sazonalidade de Vendas",
            xaxis_title="Período",
            yaxis_title="Receita (R$)",
            updatemenus=[
                dict(
                    type="buttons",
                    direction="down",
                    buttons=[
                        dict(
                            args=[{"visible": [True, False]}],
                            label="Por Dia da Semana",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [False, True]}],
                            label="Por Mês",
                            method="update"
                        )
                    ],
                    x=0.1,
                    y=1.1
                )
            ],
            showlegend=False
        )
        fig_seasonality.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_seasonality, use_container_width=True)
        
        # Rentabilidade por Segmento
        st.subheader("💰 Rentabilidade por Segmento")
        
        # Calcular rentabilidade por categoria
        category_profit = filtered_df.groupby('product_category_name').agg({
            'price': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        category_profit['avg_price'] = category_profit['price'] / category_profit['order_id']
        category_profit['profit_margin'] = 0.3  # Simulando margem de 30%
        category_profit['profit'] = category_profit['price'] * category_profit['profit_margin']
        
        # Ordenar por lucro
        category_profit = category_profit.sort_values('profit', ascending=False).head(10)
        
        # Criar gráfico de rentabilidade
        fig_profit = go.Figure()
        
        fig_profit.add_trace(go.Bar(
            x=category_profit['product_category_name'],
            y=category_profit['profit'],
            name='Lucro',
            marker_color='#2ca02c'
        ))
        
        fig_profit.update_layout(
            title="Top 10 Categorias por Rentabilidade",
            xaxis_title="Categoria",
            yaxis_title="Lucro (R$)",
            showlegend=False,
            xaxis_tickangle=45
        )
        fig_profit.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_profit, use_container_width=True)
        
        # Ticket Médio por Perfil
        st.subheader("💵 Ticket Médio por Perfil")
        
        # Calcular ticket médio por estado
        state_ticket = filtered_df.groupby('customer_state')['price'].mean().sort_values(ascending=False)
        
        # Criar gráfico de ticket médio
        fig_ticket = go.Figure()
        
        fig_ticket.add_trace(go.Bar(
            x=state_ticket.index,
            y=state_ticket.values,
            name='Ticket Médio',
            marker_color='#1f77b4'
        ))
        
        fig_ticket.update_layout(
            title="Ticket Médio por Estado",
            xaxis_title="Estado",
            yaxis_title="Ticket Médio (R$)",
            showlegend=False
        )
        fig_ticket.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_ticket, use_container_width=True)
    
    # Adicionar insights sobre os dados
    st.markdown("---")
    st.subheader("💡 Insights Principais")
    
    # Identificar o dia da semana com maior receita
    best_day = day_revenue.idxmax()
    best_day_revenue = day_revenue.max()
    
    # Identificar o mês com maior receita
    best_month = month_revenue.idxmax()
    best_month_revenue = month_revenue.max()
    
    # Identificar a categoria mais rentável
    best_category = category_profit.iloc[0]['product_category_name']
    best_category_profit = category_profit.iloc[0]['profit']
    
    # Identificar o estado com maior ticket médio
    best_state = state_ticket.idxmax()
    best_state_ticket = state_ticket.max()
    
    # Exibir insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        - **Melhor dia para vendas**: {best_day} (R$ {format_value(best_day_revenue)})
        - **Melhor mês para vendas**: {best_month} (R$ {format_value(best_month_revenue)})
        """)
    
    with col2:
        st.markdown(f"""
        - **Categoria mais rentável**: {best_category} (Lucro: R$ {format_value(best_category_profit)})
        - **Estado com maior ticket médio**: {best_state} (R$ {format_value(best_state_ticket)})
        """)
    
    # Previsão de Demanda
    st.markdown("---")
    st.subheader("📈 Previsão de Demanda por Categoria")
    
    # Calcular vendas mensais por categoria
    filtered_df['month'] = pd.to_datetime(filtered_df['order_purchase_timestamp']).dt.to_period('M')
    monthly_category_sales = filtered_df.groupby(['month', 'product_category_name'])['order_id'].count().reset_index()
    monthly_category_sales['month'] = monthly_category_sales['month'].astype(str)
    
    # Identificar as 5 categorias com maior volume de vendas
    top_categories = filtered_df.groupby('product_category_name')['order_id'].count().sort_values(ascending=False).head(5).index.tolist()
    
    # Filtrar apenas as categorias principais
    top_category_sales = monthly_category_sales[monthly_category_sales['product_category_name'].isin(top_categories)]
    
    # Criar gráfico de evolução das vendas por categoria
    fig_category_trend = px.line(
        top_category_sales,
        x='month',
        y='order_id',
        color='product_category_name',
        title="Evolução das Vendas por Categoria",
        labels={'order_id': 'Quantidade de Pedidos', 'month': 'Mês', 'product_category_name': 'Categoria'}
    )
    fig_category_trend.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig_category_trend.update_layout(dragmode=False, hovermode=False)
    st.plotly_chart(fig_category_trend, use_container_width=True)
    
    # Calcular taxa de crescimento para cada categoria
    category_growth = {}
    for category in top_categories:
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        if len(category_data) >= 2:
            first_month = category_data.iloc[0]['order_id']
            last_month = category_data.iloc[-1]['order_id']
            growth_rate = (last_month - first_month) / first_month * 100 if first_month > 0 else 0
            category_growth[category] = growth_rate
    
    # Ordenar categorias por taxa de crescimento
    sorted_categories = sorted(category_growth.items(), key=lambda x: x[1], reverse=True)
    
    # Criar gráfico de barras para taxa de crescimento
    fig_growth = go.Figure()
    
    fig_growth.add_trace(go.Bar(
        x=[cat[0] for cat in sorted_categories],
        y=[cat[1] for cat in sorted_categories],
        name='Taxa de Crescimento',
        marker_color='#2ca02c'
    ))
    
    fig_growth.update_layout(
        title="Taxa de Crescimento por Categoria",
        xaxis_title="Categoria",
        yaxis_title="Taxa de Crescimento (%)",
        showlegend=False
    )
    fig_growth.update_layout(dragmode=False, hovermode=False)
    st.plotly_chart(fig_growth, use_container_width=True)
    
    # Previsão de demanda para os próximos 3 meses
    st.subheader("🔮 Previsão de Demanda para os Próximos 3 Meses")
    
    # Criar DataFrame para previsão
    last_month = pd.to_datetime(monthly_category_sales['month'].iloc[-1])
    forecast_months = pd.date_range(start=last_month, periods=4, freq='M')[1:]
    
    # Calcular previsão para cada categoria
    forecast_data = []
    
    for category in top_categories:
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        
        if len(category_data) >= 3:
            # Calcular média móvel de 3 meses
            ma3 = category_data['order_id'].rolling(window=3).mean().iloc[-1]
            
            # Calcular tendência (últimos 3 meses)
            recent_data = category_data.tail(3)
            x = np.arange(len(recent_data))
            y = recent_data['order_id'].values
            z = np.polyfit(x, y, 1)
            trend = z[0]  # Coeficiente de crescimento mensal
            
            # Calcular previsão para os próximos 3 meses
            for i, month in enumerate(forecast_months):
                forecast = ma3 + (trend * (i + 1))
                forecast_data.append({
                    'month': month,
                    'product_category_name': category,
                    'forecast': max(0, forecast)  # Garantir que a previsão não seja negativa
                })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Criar gráfico de previsão
    fig_forecast = go.Figure()
    
    # Adicionar dados históricos
    for category in top_categories:
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        fig_forecast.add_trace(go.Scatter(
            x=category_data['month'],
            y=category_data['order_id'],
            name=f'{category} (Histórico)',
            line=dict(width=2)
        ))
    
    # Adicionar previsão
    for category in top_categories:
        category_forecast = forecast_df[forecast_df['product_category_name'] == category]
        if not category_forecast.empty:
            fig_forecast.add_trace(go.Scatter(
                x=category_forecast['month'],
                y=category_forecast['forecast'],
                name=f'{category} (Previsão)',
                line=dict(dash='dash', width=2)
            ))
    
    fig_forecast.update_layout(
        title="Previsão de Demanda para os Próximos 3 Meses",
        xaxis_title="Mês",
        yaxis_title="Quantidade de Pedidos",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_forecast.update_layout(dragmode=False, hovermode=False)
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Exibir recomendações baseadas na previsão
    st.subheader("💡 Recomendações de Estoque")
    
    # Calcular variação percentual entre último mês real e previsão para o próximo mês
    recommendations = []
    
    for category in top_categories:
        category_data = top_category_sales[top_category_sales['product_category_name'] == category]
        category_forecast = forecast_df[forecast_df['product_category_name'] == category]
        
        if not category_data.empty and not category_forecast.empty:
            last_month_sales = category_data.iloc[-1]['order_id']
            next_month_forecast = category_forecast.iloc[0]['forecast']
            
            variation = (next_month_forecast - last_month_sales) / last_month_sales * 100 if last_month_sales > 0 else 0
            
            if variation > 20:
                action = "Aumentar significativamente"
            elif variation > 10:
                action = "Aumentar"
            elif variation < -20:
                action = "Reduzir significativamente"
            elif variation < -10:
                action = "Reduzir"
            else:
                action = "Manter"
                
            recommendations.append({
                'category': category,
                'variation': variation,
                'action': action
            })
    
    # Ordenar recomendações por variação absoluta
    recommendations.sort(key=lambda x: abs(x['variation']), reverse=True)
    
    # Exibir recomendações
    for rec in recommendations:
        st.markdown(f"**{rec['category']}**: {rec['action']} o estoque (variação prevista: {format_value(rec['variation'])}%)")

elif pagina == "Aquisição e Retenção":
    st.title("Aquisição e Retenção")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    acquisition_kpis = calculate_acquisition_retention_kpis(filtered_df, marketing_spend, date_range)
    
    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)
    
    # Primeira linha de KPIs
    col1.metric("👥 Novos Clientes (Período)", format_value(acquisition_kpis['total_new_customers'], is_integer=True))
    col2.metric("🔄 Taxa de Recompra", format_percentage(acquisition_kpis['repurchase_rate']))
    col3.metric("⏳ Tempo até 2ª Compra", f"{int(acquisition_kpis['avg_time_to_second'])} dias")
    
    # Segunda linha de KPIs
    col1.metric("💰 CAC", f"R$ {format_value(acquisition_kpis['cac'])}")
    col2.metric("🔁 LTV", f"R$ {format_value(acquisition_kpis['ltv'])}")
    col3.metric("📈 LTV/CAC", format_value(acquisition_kpis['ltv'] / acquisition_kpis['cac'] if acquisition_kpis['cac'] > 0 else 0))
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Novos vs Retornando
        st.subheader("👥 Novos vs Clientes Retornando")
        fig_customers = go.Figure()
        
        # Adicionar novos clientes
        fig_customers.add_trace(go.Bar(
            x=acquisition_kpis['new_customers']['month'],
            y=acquisition_kpis['new_customers']['customer_unique_id'],
            name='Novos Clientes',
            marker_color='#1f77b4'
        ))
        
        # Adicionar clientes retornando
        fig_customers.add_trace(go.Bar(
            x=acquisition_kpis['returning_customers']['month'],
            y=acquisition_kpis['returning_customers']['customer_unique_id'],
            name='Clientes Retornando',
            marker_color='#2ca02c'
        ))
        
        fig_customers.update_layout(
            title="Evolução de Novos e Clientes Retornando",
            barmode='stack',
            xaxis_title="Mês",
            yaxis_title="Número de Clientes",
            yaxis=dict(tickformat=",d")
        )
        fig_customers.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_customers, use_container_width=True)
        
        # Gráfico de Evolução dos Novos Clientes
        st.subheader("📈 Evolução dos Novos Clientes")
        fig_new = px.line(acquisition_kpis['new_customers'],
                         x='month',
                         y='customer_unique_id',
                         title="Evolução Mensal de Novos Clientes",
                         labels={'customer_unique_id': 'Novos Clientes', 'month': 'Mês'})
        fig_new.update_layout(
            yaxis=dict(tickformat=",d"),
            showlegend=False
        )
        fig_new.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_new, use_container_width=True)
    
    with col2:
        # Funil de Conversão
        st.subheader("🔄 Funil de Conversão")
        fig_funnel = go.Figure(go.Funnel(
            y=acquisition_kpis['funnel_data']['Etapa'],
            x=acquisition_kpis['funnel_data']['Quantidade'],
            textinfo="value+percent initial",
            textposition="inside",
            marker=dict(color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ))
        fig_funnel.update_layout(
            title="Funil de Conversão",
            showlegend=False
        )
        fig_funnel.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_funnel, use_container_width=True)
        
        # Comparativo LTV vs CAC
        st.subheader("💰 Evolução LTV vs CAC")
        
        # Calcular LTV e CAC por mês
        monthly_metrics = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M')).agg({
            'price': 'sum',
            'customer_unique_id': 'nunique',
            'pedido_cancelado': 'sum'
        }).reset_index()
        
        monthly_metrics['order_purchase_timestamp'] = monthly_metrics['order_purchase_timestamp'].astype(str)
        monthly_metrics['monthly_revenue'] = monthly_metrics['price'] - (monthly_metrics['price'] * monthly_metrics['pedido_cancelado'])
        monthly_metrics['monthly_ltv'] = monthly_metrics['monthly_revenue'] / monthly_metrics['customer_unique_id']
        monthly_metrics['monthly_cac'] = marketing_spend / 12  # Distribuindo o gasto mensalmente
        monthly_metrics['ltv_cac_ratio'] = monthly_metrics['monthly_ltv'] / monthly_metrics['monthly_cac']
        
        # Inverter o sinal do LTV
        monthly_metrics['monthly_ltv'] = -monthly_metrics['monthly_ltv']
        
        # Criar gráfico de área
        fig_comparison = go.Figure()
        
        # Adicionar área do LTV
        fig_comparison.add_trace(go.Scatter(
            x=monthly_metrics['order_purchase_timestamp'],
            y=monthly_metrics['monthly_ltv'],
            name='LTV',
            fill='tozeroy',
            line=dict(color='rgba(46, 204, 113, 0.3)'),
            fillcolor='rgba(46, 204, 113, 0.3)'
        ))
        
        # Adicionar área do CAC
        fig_comparison.add_trace(go.Scatter(
            x=monthly_metrics['order_purchase_timestamp'],
            y=monthly_metrics['monthly_cac'],
            name='CAC',
            fill='tozeroy',
            line=dict(color='rgba(231, 76, 60, 0.3)'),
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        
        # Adicionar linha da razão LTV/CAC
        fig_comparison.add_trace(go.Scatter(
            x=monthly_metrics['order_purchase_timestamp'],
            y=monthly_metrics['ltv_cac_ratio'],
            name='Razão LTV/CAC',
            line=dict(color='#2c3e50', width=2),
            yaxis='y2'
        ))
        
        # Configurar layout
        fig_comparison.update_layout(
            title="Evolução do LTV vs CAC ao Longo do Tempo",
            xaxis_title="Mês",
            yaxis=dict(
                title="Valor (R$)",
                side="left"
            ),
            yaxis2=dict(
                title="Razão LTV/CAC",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Adicionar linhas de referência usando shapes
        fig_comparison.add_shape(
            type="line",
            x0=monthly_metrics['order_purchase_timestamp'].iloc[0],
            x1=monthly_metrics['order_purchase_timestamp'].iloc[-1],
            y0=0,
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            yref="y"
        )
        
        fig_comparison.add_shape(
            type="line",
            x0=monthly_metrics['order_purchase_timestamp'].iloc[0],
            x1=monthly_metrics['order_purchase_timestamp'].iloc[-1],
            y0=1,
            y1=1,
            line=dict(color="gray", width=1, dash="dash"),
            yref="y2"
        )
        
        fig_comparison.add_shape(
            type="line",
            x0=monthly_metrics['order_purchase_timestamp'].iloc[0],
            x1=monthly_metrics['order_purchase_timestamp'].iloc[-1],
            y0=3,
            y1=3,
            line=dict(color="green", width=1, dash="dash"),
            yref="y2"
        )
        
        fig_comparison.update_layout(dragmode=False)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Adicionar interpretação da razão LTV/CAC
        st.markdown("""
        ### 📊 Interpretação da Razão LTV/CAC
        
        A razão LTV/CAC (Lifetime Value / Customer Acquisition Cost) indica quanto valor um cliente traz ao longo do tempo comparado ao custo de adquiri-lo.
        
        | Razão LTV/CAC | Interpretação | Situação |
        |--------------|---------------|----------|
        | < 1 | Você perde dinheiro por cliente | 🚨 Ruim. Custa mais do que retorna. |
        | = 1 | Você empata | ⚠️ Não é sustentável. |
        | 1 < x < 3 | Lucro baixo | 😬 Razoável, mas pode melhorar. |
        | = 3 | Ponto ideal (clássico) | ✅ Saudável, lucro balanceado. |
        | > 3 | Lucro alto | 💰 Pode ser bom... ou pode estar subinvestindo. |
        
        **Por que é crítica?**
        - Ajuda a decidir quanto investir em marketing com segurança
        - Mostra eficiência do funil de aquisição e retenção
        - Indica se o negócio está sustentável no longo prazo
        - Métrica-chave para investidores
        """)
        
        # Calcular e exibir métricas atuais de LTV/CAC
        current_ltv = acquisition_kpis['ltv']
        current_cac = acquisition_kpis['cac']
        current_ratio = current_ltv / current_cac if current_cac > 0 else 0
        
        # Determinar status com base na razão atual
        if current_ratio < 1:
            status = "🚨 Crítico"
            status_color = "red"
        elif current_ratio == 1:
            status = "⚠️ Limite"
            status_color = "orange"
        elif current_ratio < 3:
            status = "😬 Razoável"
            status_color = "yellow"
        elif current_ratio == 3:
            status = "✅ Ideal"
            status_color = "green"
        else:
            status = "💰 Alto"
            status_color = "blue"
        
        # Exibir métricas atuais
        st.subheader("📈 Métricas Atuais de LTV/CAC")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("LTV", f"R$ {format_value(current_ltv)}")
        col2.metric("CAC", f"R$ {format_value(current_cac)}")
        col3.metric("Razão LTV/CAC", format_value(current_ratio))
        col4.markdown(f"<h3 style='color: {status_color};'>{status}</h3>", unsafe_allow_html=True)
        
        # Análise de tendência
        if len(monthly_metrics) >= 3:
            recent_ratio = monthly_metrics['ltv_cac_ratio'].tail(3).mean()
            older_ratio = monthly_metrics['ltv_cac_ratio'].head(3).mean()
            trend = (recent_ratio - older_ratio) / older_ratio * 100 if older_ratio > 0 else 0
            
            st.subheader("📈 Análise de Tendência")
            
            if trend > 10:
                st.success(f"A razão LTV/CAC está em tendência de **alta** (+{format_value(trend)}% nos últimos 3 meses). Isso indica que a eficiência de aquisição de clientes está melhorando.")
            elif trend < -10:
                st.error(f"A razão LTV/CAC está em tendência de **baixa** ({format_value(trend)}% nos últimos 3 meses). Isso indica que a eficiência de aquisição de clientes está piorando.")
            else:
                st.info(f"A razão LTV/CAC está **estável** ({format_value(trend)}% nos últimos 3 meses).")
        
        # Recomendações baseadas na razão LTV/CAC
        st.subheader("💡 Recomendações")
        
        if current_ratio < 1:
            st.markdown("""
            - **Reduzir o CAC**: Otimize suas campanhas de marketing para reduzir o custo de aquisição
            - **Aumentar o LTV**: Implemente estratégias de upselling e cross-selling para aumentar o valor dos clientes
            - **Revisar o modelo de negócio**: Avalie se o preço dos produtos/serviços está adequado
            """)
        elif current_ratio < 3:
            st.markdown("""
            - **Testar novos canais de aquisição**: Explore canais com potencial de menor CAC
            - **Melhorar a retenção**: Implemente programas de fidelidade para aumentar o LTV
            - **Otimizar o funil de conversão**: Identifique e corrija gargalos no processo de aquisição
            """)
        elif current_ratio > 5:
            st.markdown("""
            - **Aumentar investimento em marketing**: Você pode estar subinvestindo em crescimento
            - **Expandir para novos mercados**: Aproveite a eficiência atual para escalar o negócio
            - **Diversificar canais de aquisição**: Explore novos canais para manter a eficiência
            """)
        else:
            st.markdown("""
            - **Manter o equilíbrio atual**: Continue monitorando a razão LTV/CAC
            - **Testar pequenos aumentos no CAC**: Experimente aumentar o investimento em marketing para ver se mantém a eficiência
            - **Focar em melhorias incrementais**: Pequenas otimizações podem levar a ganhos significativos
            """)

elif pagina == "Comportamento do Cliente":
    st.title("Comportamento do Cliente")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    acquisition_kpis = calculate_acquisition_retention_kpis(filtered_df, marketing_spend, date_range)
    
    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)
    
    # Primeira linha de KPIs
    col1.metric("🎯 Taxa de Abandono", format_percentage(kpis['abandonment_rate']))
    col2.metric("😊 Satisfação do Cliente", format_value(kpis['csat']))
    col3.metric("💰 Ticket Médio", f"R$ {format_value(kpis['average_ticket'])}")
    
    # Segunda linha de KPIs
    col1.metric("📦 Tempo Médio de Entrega", f"{int(kpis['avg_delivery_time'])} dias")
    col2.metric("🔄 Taxa de Recompra", format_percentage(acquisition_kpis['repurchase_rate']))
    col3.metric("⏳ Tempo até 2ª Compra", f"{int(acquisition_kpis['avg_time_to_second'])} dias")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Satisfação do Cliente ao Longo do Tempo
        st.subheader("😊 Satisfação do Cliente ao Longo do Tempo")
        satisfaction_data = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['review_score'].mean().reset_index()
        satisfaction_data['order_purchase_timestamp'] = satisfaction_data['order_purchase_timestamp'].astype(str)
        fig_satisfaction = px.line(
            satisfaction_data,
            x='order_purchase_timestamp',
            y='review_score',
            title="Evolução da Satisfação do Cliente",
            labels={'review_score': 'Nota Média', 'order_purchase_timestamp': 'Mês'}
        )
        fig_satisfaction.update_layout(
            yaxis=dict(range=[0, 5]),
            showlegend=False
        )
        fig_satisfaction.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_satisfaction, use_container_width=True)
        
        # Gráfico de Distribuição de Satisfação
        st.subheader("📊 Distribuição de Satisfação")
        fig_dist = px.histogram(
            filtered_df,
            x='review_score',
            title="Distribuição das Avaliações",
            labels={'review_score': 'Nota', 'count': 'Quantidade de Avaliações'}
        )
        fig_dist.update_layout(
            xaxis=dict(range=[0, 5]),
            showlegend=False
        )
        fig_dist.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Gráfico de Tempo de Entrega ao Longo do Tempo
        st.subheader("📦 Tempo de Entrega ao Longo do Tempo")
        filtered_df['delivery_time'] = (pd.to_datetime(filtered_df['order_delivered_customer_date']) - 
                             pd.to_datetime(filtered_df['order_purchase_timestamp'])).dt.days
        delivery_data = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['delivery_time'].mean().reset_index()
        delivery_data['order_purchase_timestamp'] = delivery_data['order_purchase_timestamp'].astype(str)
        fig_delivery = px.line(
            delivery_data,
            x='order_purchase_timestamp',
            y='delivery_time',
            title="Evolução do Tempo de Entrega",
            labels={'delivery_time': 'Dias', 'order_purchase_timestamp': 'Mês'}
        )
        fig_delivery.update_layout(showlegend=False)
        fig_delivery.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_delivery, use_container_width=True)
        
        # Gráfico de Ticket Médio ao Longo do Tempo
        st.subheader("💰 Ticket Médio ao Longo do Tempo")
        ticket_data = filtered_df.groupby(filtered_df['order_purchase_timestamp'].dt.to_period('M'))['price'].mean().reset_index()
        ticket_data['order_purchase_timestamp'] = ticket_data['order_purchase_timestamp'].astype(str)
        fig_ticket = px.line(
            ticket_data,
            x='order_purchase_timestamp',
            y='price',
            title="Evolução do Ticket Médio",
            labels={'price': 'Valor Médio (R$)', 'order_purchase_timestamp': 'Mês'}
        )
        fig_ticket.update_layout(showlegend=False)
        fig_ticket.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_ticket, use_container_width=True)

elif pagina == "Produtos e Categorias":
    st.title("Produtos e Categorias")
    kpis = calculate_kpis(filtered_df, marketing_spend, date_range)
    
    # Layout dos KPIs
    col1, col2, col3 = st.columns(3)
    
    # Primeira linha de KPIs
    col1.metric("📦 Total de Produtos", format_value(kpis['total_products'], is_integer=True))
    col2.metric("🏷️ Categorias Únicas", format_value(kpis['unique_categories'], is_integer=True))
    col3.metric("💰 Ticket Médio", f"R$ {format_value(kpis['average_ticket'])}")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Categorias por Receita
        st.subheader("💰 Top 10 Categorias por Receita")
        category_revenue = filtered_df.groupby('product_category_name')['price'].sum().sort_values(ascending=False).head(10)
        fig_category = px.bar(
            x=category_revenue.index,
            y=category_revenue.values,
            title="Top 10 Categorias por Receita",
            labels={'x': 'Categoria', 'y': 'Receita (R$)'}
        )
        fig_category.update_layout(showlegend=False)
        fig_category.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Top 10 Categorias por Quantidade
        st.subheader("📦 Top 10 Categorias por Quantidade")
        category_quantity = filtered_df.groupby('product_category_name')['order_id'].count().sort_values(ascending=False).head(10)
        fig_quantity = px.bar(
            x=category_quantity.index,
            y=category_quantity.values,
            title="Top 10 Categorias por Quantidade",
            labels={'x': 'Categoria', 'y': 'Quantidade de Pedidos'}
        )
        fig_quantity.update_layout(showlegend=False)
        fig_quantity.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_quantity, use_container_width=True)
    
    with col2:
        # Distribuição de Preços por Categoria
        st.subheader("💵 Distribuição de Preços por Categoria")
        fig_price_dist = px.box(
            filtered_df,
            x='product_category_name',
            y='price',
            title="Distribuição de Preços por Categoria",
            labels={'price': 'Preço (R$)', 'product_category_name': 'Categoria'}
        )
        fig_price_dist.update_layout(showlegend=False)
        fig_price_dist.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_price_dist, use_container_width=True)
        
        # Taxa de Cancelamento por Categoria
        st.subheader("❌ Taxa de Cancelamento por Categoria")
        category_cancellation = filtered_df.groupby('product_category_name')['pedido_cancelado'].mean().sort_values(ascending=False)
        fig_cancellation = px.bar(
            x=category_cancellation.index,
            y=category_cancellation.values,
            title="Taxa de Cancelamento por Categoria",
            labels={'x': 'Categoria', 'y': 'Taxa de Cancelamento'}
        )
        fig_cancellation.update_layout(
            yaxis=dict(tickformat=".1%"),
            showlegend=False
        )
        fig_cancellation.update_layout(dragmode=False, hovermode=False)
        st.plotly_chart(fig_cancellation, use_container_width=True)
