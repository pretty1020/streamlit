
library(rsconnect)


rsconnect::setAccountInfo(name='marian1020',
                          token='8F00BE4117C348843CC9FA257273DBA5',
                          secret='HMHScd8drDmRkdsgsvt9BUPMPMDgAtbjkvBoUVun')



library(shiny)

##### ------------- Erlang C Functions ------------- #####

erlang_c <- function(traffic, agents) {
  if (agents <= traffic) return(1)
  rho <- traffic / agents
  
  sum_terms <- sapply(0:(agents - 1), function(k) traffic^k / factorial(k))
  last_term <- (traffic^agents / factorial(agents)) * (1 / (1 - rho))
  denom <- sum(sum_terms) + last_term
  
  P0 <- 1 / denom
  Pw <- last_term * P0
  Pw
}

erlang_c_service_level <- function(traffic, agents, target_answer_time, aht) {
  if (agents <= traffic) return(0)
  Pw <- erlang_c(traffic, agents)
  1 - Pw * exp(-(agents - traffic) * (target_answer_time / aht))
}

erlang_c_asa <- function(traffic, agents, aht) {
  if (agents <= traffic) return(Inf)
  Pw <- erlang_c(traffic, agents)
  (Pw * aht) / (agents - traffic)
}

find_required_agents <- function(
    calls,
    aht_sec,
    interval_minutes,
    target_sl,
    target_answer_time_sec,
    max_agents = 1000
) {
  interval_sec <- interval_minutes * 60
  lambda <- calls / interval_sec
  traffic <- lambda * aht_sec
  
  if (traffic <= 0) return(NULL)
  
  start_agents <- max(1, ceiling(traffic) + 1)
  
  for (N in start_agents:max_agents) {
    SL  <- erlang_c_service_level(traffic, N, target_answer_time_sec, aht_sec)
    ASA <- erlang_c_asa(traffic, N, aht_sec)
    occ <- traffic / N
    
    if (SL >= target_sl) {
      return(list(
        agents        = N,
        service_level = SL,
        asa           = ASA,
        occupancy     = occ,
        traffic       = traffic
      ))
    }
  }
  NULL
}

##### ------------- UI ------------- #####

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        background-color: #0f172a;
        color: #e5e7eb;
        font-family: 'Segoe UI', system-ui, sans-serif;
      }
      .well {
        background-color: #020617;
        border-radius: 16px;
        border: 1px solid #1f2937;
      }
      .panel {
        background-color: #020617;
        border-radius: 16px;
        border: 1px solid #1f2937;
      }
      h2, h3, h4, h5 {
        color: #f9fafb;
      }
      .calc-title {
        font-weight: 700;
        font-size: 26px;
        margin-bottom: 4px;
      }
      .calc-subtitle {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 16px;
      }
      .result-card {
        background: radial-gradient(circle at top left, #6366f1 0, #020617 45%);
        border-radius: 20px;
        padding: 18px;
        margin-bottom: 16px;
        color: #e5e7eb;
      }
      .result-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #cbd5f5;
      }
      .result-value {
        font-size: 28px;
        font-weight: 700;
      }
      .result-unit {
        font-size: 13px;
        color: #cbd5f5;
      }
      .info-text {
        font-size: 13px;
        color: #9ca3af;
      }
      .shiny-input-container {
        margin-bottom: 8px;
      }
    "))
  ),
  titlePanel(NULL),
  
  tabsetPanel(
    id = "main_tabs",
    
    #### TAB 1 – WFM CALCULATOR ####
    tabPanel(
      "WFM Calculator",
      br(),
      fluidRow(
        column(
          width = 4,
          div(class = "well",
              div(class = "calc-title", "Workforce Management Calculator"),
              div(class = "calc-subtitle",
                  "Quickly estimate required agents, FTE, and headcount using Erlang C."
              ),
              
              h4("Contact & Handle Time"),
              numericInput("calls", "Calls per Interval",
                           value = 300, min = 0, step = 1),
              numericInput("aht", "Average Handle Time (seconds)",
                           value = 300, min = 1, step = 1),
              numericInput("interval", "Interval Length (minutes)",
                           value = 30, min = 1, step = 1),
              
              tags$hr(),
              h4("Service Level Targets"),
              sliderInput("target_sl", "Target Service Level (%)",
                          min = 50, max = 99, value = 80, step = 1),
              numericInput("target_t", "Target Answer Time (seconds)",
                           value = 20, min = 1, step = 1),
              
              tags$hr(),
              h4("Planning Assumptions"),
              sliderInput("shrinkage", "Shrinkage (%)",
                          min = 0, max = 60, value = 30, step = 1),
              numericInput("fte_hours", "FTE Hours per Week",
                           value = 40, min = 1, step = 1),
              numericInput("hours_per_agent",
                           "Planned Productive Hours per Agent per Week",
                           value = 40, min = 1, step = 1),
              
              tags$hr(),
              p(class = "info-text",
                "Tip: Change calls and AHT to simulate peaks, promo days, or holidays.")
          )
        ),
        
        column(
          width = 8,
          fluidRow(
            column(
              width = 6,
              div(class = "result-card",
                  div(class = "result-label", "Required Agents (Interval)"),
                  htmlOutput("agents_required"),
                  div(class = "info-text",
                      "Minimum number of agents to hit your target service level for this interval.")
              )
            ),
            column(
              width = 6,
              div(class = "result-card",
                  div(class = "result-label", "Net & Planned FTE"),
                  htmlOutput("fte_required")
              )
            )
          ),
          fluidRow(
            column(
              width = 12,
              div(class = "panel panel-default",
                  div(class = "panel-heading",
                      h4("Queue Performance Metrics (Erlang C)")
                  ),
                  div(class = "panel-body",
                      tableOutput("metrics_table"),
                      tags$hr(),
                      h5("Explanation"),
                      htmlOutput("interpretation")
                  )
              )
            )
          )
        )
      )
    ),
    
    #### TAB 2 – FORECASTING TOOL ####
    tabPanel(
      "Forecasting Tool",
      br(),
      fluidRow(
        column(
          width = 4,
          div(class = "well",
              div(class = "calc-title", "Simple Volume Forecast"),
              div(class = "calc-subtitle",
                  "Forecast call volume and see the impact on required agents."
              ),
              numericInput("fc_base_calls", "Base Calls per Interval",
                           value = 300, min = 0, step = 1),
              sliderInput("fc_pct_change", "Forecast Change in Calls (%)",
                          min = -50, max = 200, value = 0, step = 5),
              p(class = "info-text",
                "Negative values = drop in volume; positive = growth."),
              
              tags$hr(),
              h4("Forecast Scenario Settings"),
              numericInput("fc_aht", "Average Handle Time (seconds)",
                           value = 300, min = 1, step = 1),
              numericInput("fc_interval", "Interval Length (minutes)",
                           value = 30, min = 1, step = 1),
              sliderInput("fc_target_sl", "Target Service Level (%)",
                          min = 50, max = 99, value = 80, step = 1),
              numericInput("fc_target_t", "Target Answer Time (seconds)",
                           value = 20, min = 1, step = 1)
          )
        ),
        column(
          width = 8,
          fluidRow(
            column(
              width = 6,
              div(class = "result-card",
                  div(class = "result-label", "Forecasted Calls per Interval"),
                  htmlOutput("fc_calls_out"),
                  div(class = "info-text",
                      "Based on base volume and % change.")
              )
            ),
            column(
              width = 6,
              div(class = "result-card",
                  div(class = "result-label", "Required Agents (Forecast)"),
                  htmlOutput("fc_agents_out"),
                  div(class = "info-text",
                      "Agents needed under this forecasted volume.")
              )
            )
          ),
          fluidRow(
            column(
              width = 12,
              div(class = "panel panel-default",
                  div(class = "panel-heading",
                      h4("Forecast Scenario Details")
                  ),
                  div(class = "panel-body",
                      tableOutput("fc_metrics_table"),
                      tags$hr(),
                      htmlOutput("fc_explanation")
                  )
              )
            )
          )
        )
      )
    )
  )
)

##### ------------- SERVER ------------- #####

server <- function(input, output, session) {
  
  #### MAIN WFM CALCULATOR ####
  calc_results <- reactive({
    res <- find_required_agents(
      calls                  = input$calls,
      aht_sec                = input$aht,
      interval_minutes       = input$interval,
      target_sl              = input$target_sl / 100,
      target_answer_time_sec = input$target_t
    )
    
    if (is.null(res)) return(NULL)
    
    agents  <- res$agents
    net_fte <- agents * (input$hours_per_agent / input$fte_hours)
    
    shrinkage_factor <- 1 - (input$shrinkage / 100)
    if (shrinkage_factor <= 0) shrinkage_factor <- 0.01
    planned_fte <- net_fte / shrinkage_factor
    
    list(
      agents        = agents,
      service_level = res$service_level,
      asa           = res$asa,
      occupancy     = res$occupancy,
      traffic       = res$traffic,
      net_fte       = net_fte,
      planned_fte   = planned_fte
    )
  })
  
  output$agents_required <- renderUI({
    res <- calc_results()
    if (is.null(res)) {
      return(HTML("<span class='result-value'>Not available</span>"))
    }
    HTML(paste0(
      "<span class='result-value'>",
      format(round(res$agents, 0), big.mark = ","),
      "</span> <span class='result-unit'>agents</span>"
    ))
  })
  
  output$fte_required <- renderUI({
    res <- calc_results()
    if (is.null(res)) {
      HTML("<span class='result-value'>–</span>")
    } else {
      HTML(paste0(
        "<div>",
        "<span class='result-label'>Net FTE</span><br>",
        "<span class='result-value'>", round(res$net_fte, 2), "</span>",
        "<span class='result-unit'> FTE</span>",
        "</div><br>",
        "<div>",
        "<span class='result-label'>Planned FTE (with shrinkage)</span><br>",
        "<span class='result-value'>", round(res$planned_fte, 2), "</span>",
        "<span class='result-unit'> FTE</span>",
        "</div>"
      ))
    }
  })
  
  output$metrics_table <- renderTable({
    res <- calc_results()
    if (is.null(res)) {
      return(data.frame(
        Metric = c("Traffic (Erlangs)", "Service Level", "ASA (sec)", "Occupancy"),
        Value  = c("N/A", "N/A", "N/A", "N/A")
      ))
    }
    
    data.frame(
      Metric = c(
        "Traffic (Erlangs)",
        "Service Level (%)",
        "Average Speed of Answer (sec)",
        "Occupancy (%)"
      ),
      Value = c(
        round(res$traffic, 3),
        round(res$service_level * 100, 2),
        round(res$asa, 2),
        round(res$occupancy * 100, 2)
      ),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  
  output$interpretation <- renderUI({
    res <- calc_results()
    if (is.null(res)) {
      return(HTML(
        "Enter your call forecast and AHT on the left. The calculator will estimate required agents and FTE to hit your target service level."
      ))
    }
    
    sl     <- round(res$service_level * 100, 1)
    asa    <- round(res$asa, 1)
    occ    <- round(res$occupancy * 100, 1)
    agents <- res$agents
    shrink <- input$shrinkage
    
    HTML(paste0(
      "With <b>", input$calls, "</b> calls every <b>", input$interval,
      " minutes</b> and an AHT of <b>", input$aht, " seconds</b>, you need approximately ",
      "<b>", agents, " agents</b> logged in to achieve around <b>", sl,
      "%</b> service level (target: ", input$target_sl,
      "%) with an ASA of about <b>", asa, " seconds</b>. ",
      "Estimated occupancy is <b>", occ,
      "%</b>. After applying <b>", shrink, "%</b> shrinkage, this translates to roughly <b>",
      round(res$planned_fte, 2), " FTE</b> in your weekly plan."
    ))
  })
  
  #### FORECASTING TOOL ####
  forecast_results <- reactive({
    eff_calls <- input$fc_base_calls * (1 + input$fc_pct_change / 100)
    eff_calls <- max(eff_calls, 0)
    
    res <- find_required_agents(
      calls                  = eff_calls,
      aht_sec                = input$fc_aht,
      interval_minutes       = input$fc_interval,
      target_sl              = input$fc_target_sl / 100,
      target_answer_time_sec = input$fc_target_t
    )
    
    if (is.null(res)) return(NULL)
    
    list(
      eff_calls    = eff_calls,
      agents       = res$agents,
      service_level = res$service_level,
      asa          = res$asa,
      occupancy    = res$occupancy,
      traffic      = res$traffic
    )
  })
  
  output$fc_calls_out <- renderUI({
    res <- forecast_results()
    if (is.null(res)) {
      return(HTML("<span class='result-value'>N/A</span>"))
    }
    HTML(paste0(
      "<span class='result-value'>",
      format(round(res$eff_calls, 0), big.mark = ","),
      "</span> <span class='result-unit'>calls / interval</span>"
    ))
  })
  
  output$fc_agents_out <- renderUI({
    res <- forecast_results()
    if (is.null(res)) {
      return(HTML("<span class='result-value'>Not available</span>"))
    }
    HTML(paste0(
      "<span class='result-value'>",
      format(round(res$agents, 0), big.mark = ","),
      "</span> <span class='result-unit'>agents</span>"
    ))
  })
  
  output$fc_metrics_table <- renderTable({
    res <- forecast_results()
    if (is.null(res)) {
      return(data.frame(
        Metric = c("Traffic (Erlangs)", "Service Level", "ASA (sec)", "Occupancy"),
        Value  = c("N/A", "N/A", "N/A", "N/A")
      ))
    }
    
    data.frame(
      Metric = c(
        "Traffic (Erlangs)",
        "Service Level (%)",
        "Average Speed of Answer (sec)",
        "Occupancy (%)"
      ),
      Value = c(
        round(res$traffic, 3),
        round(res$service_level * 100, 2),
        round(res$asa, 2),
        round(res$occupancy * 100, 2)
      ),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  })
  
  output$fc_explanation <- renderUI({
    res <- forecast_results()
    if (is.null(res)) {
      return(HTML(
        "Set a base volume and a % change to see the forecasted calls and estimated agents needed for that scenario."
      ))
    }
    
    sl  <- round(res$service_level * 100, 1)
    asa <- round(res$asa, 1)
    occ <- round(res$occupancy * 100, 1)
    
    HTML(paste0(
      "Starting from <b>", input$fc_base_calls, "</b> calls per interval and a change of <b>",
      input$fc_pct_change, "%</b>, you are planning for about <b>",
      round(res$eff_calls, 0), " calls per interval</b>. ",
      "To meet a <b>", input$fc_target_sl, "%</b> service level with an AHT of <b>",
      input$fc_aht, " seconds</b>, you need approximately <b>",
      res$agents, " agents</b>. This yields roughly <b>",
      sl, "%</b> service level, ASA around <b>", asa,
      " seconds</b>, and occupancy of <b>", occ, "%</b>."
    ))
  })
}

##### ------------- RUN APP ------------- #####

shinyApp(ui = ui, server = server)
