import streamlit as st
from fontTools.misc.timeTools import MONTHNAMES
from gurobipy import Model, GRB, quicksum
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt


# Load data from JSON file
with open("data.json", "r") as file:
    data = json.load(file)

#Data Handling
months = range(0, 7)
products = data["products"]
demand = data["demand"]
time_req = data["time_required"]
monthname = {i + 1: data["months"][i] for i in range(len(data["months"]))}
profit = {i + 1: data["profit"][i] for i in range(len(data["profit"]))}
name_to_abbreviation = {"Grinding": "GR", "VerticalDrilling": "VD", "HorizontalDrilling": "HD", "Boring": "BR", "Planing": "PL"}
machine_names = {"GR": "Grinder", "VD": "Vertical Drill", "HD": "Horizontal Drill", "BR": "Borer", "PL":  "Planer"}
processing_time = {name_to_abbreviation[key]: {i + 1: time_req[key][i] for i in range(len(time_req[key]))} for key in time_req}
machine_types = {name_to_abbreviation.get(key, key): value for key, value in data["machine_types"].items()}
machine_availability = {name_to_abbreviation[key]: value for key, value in data["machine_availability"].items()}
initial_inventory=data["initial_inventory"]
final_inventory=data["final_inventory"]
working_hours_per_day = data["working_hours_per_day"]
working_days_per_month = data["working_days_per_month"]
total_hours_per_machine = working_hours_per_day * working_days_per_month
market_demand = {}
for index, row in enumerate(demand, start=1):
    market_demand[index] = {i: row[i] for i in range(len(row))}



#**********************************************#
# === Streamlit Layout ===
#**********************************************#
st.set_page_config(page_title="Factory Production Optimization", page_icon="🏭", layout="wide")

st.markdown("""
    <style>
        .title-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f4f4f9;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            color: #333;
        }
        .title-bar h1 {
            margin: 0;
            font-size: 49px;
            color: #333;
            margin-left: 20px;
        }
        .title-bar .logo1 {
            max-width: auto;
            height: 60px;
            margin-right: 20px;
        }
        .title-bar a {
            text-decoration: none;
            color: #0073e6;
            font-size: 16px;
        }
        .footer-text {
            font-size: 20px;
            background-color: #f4f4f9;
            text-align: left;
            color: #333;
            border-bottom-left-radius: 5px;
            border-bottom-right-radius: 5px;
        }
    </style>
    <div class="title-bar">
        <h1>Problem 12.3 <br> Factory Planning</h1>
        <div>
            <a href="https://decisionopt.com" target="_blank">
                <img src="https://decisionopt.com/static/media/DecisionOptLogo.7023a4e646b230de9fb8ff2717782860.svg" class="logo1" alt="Logo"/>
            </a>
        </div>
    </div>
    <div class="footer-text">
    <p style="margin-left:20px;">  'Model Building in Mathematical Programming, Fifth Edition' by H. Paul Williams</p>
    </div>    
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .container-c1 p {
            font-size: 20px;
        }
        .button {
            background-color: #FFFFFF;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 4px;
        }
        .button:hover {
            background-color: #FFFFFF;
             box-shadow: 1px 1px 4px rgb(255, 75, 75); /* Shadow effect on hover */
        }
    </style>
    <div class="container-c1">
        <br><p> For a detailed view of the mathematical formulation, please visit my 
        <a href="https://github.com/Ash7erix/Model_Building_Assignments/tree/main/12.4_Factory_Planning_Continued">Github</a> page.</p>

    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .container-c1 p {
            font-size: 20px;
        }
    </style>
    <div class="container-c1">
        <br><p>This app optimizes the factory production process by determining the optimal quantity of 
        products to manufacture, store, and sell each month, maximizing overall profit. It utilizes 
        <b>Gurobi</b> for optimization, considering factors such as production capacity, market demand, 
        inventory limits, and storage costs.</p>  
        <br><p>You can customize key parameters like storage costs and production limits using the options on 
        the left side. The app provides detailed insights, including monthly production, previously held stock, 
        sold quantities, and remaining inventory, helping you make data-driven decisions.</p>
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .container-c1 p {
            font-size: 20px;
        }
    </style>
    <div class="container-c1">
        <br><p> You can view the mathematical formulation below by clicking the button.</p>
    </div>
""", unsafe_allow_html=True)
if st.button('Display Formulation'):
    def fetch_readme(repo_url):
        raw_url = f"{repo_url}/raw/main/12.4_Factory_Planning_Continued/README.md"  # Adjust path if necessary
        response = requests.get(raw_url)
        return response.text
    repo_url = "https://github.com/Ash7erix/Model_Building_Assignments"
    try:
        readme_content = fetch_readme(repo_url)
        st.markdown(readme_content)
        st.markdown("""---""")
    except Exception as e:
        st.error(f"Could not fetch README: {e}")
        st.markdown("""---""")

st.title("Optimization Data and Constraints:")
st.sidebar.header("Optimization Parameters")
storage_cost = st.sidebar.number_input("Storage Cost per Unit (£)", min_value=0.0, value=0.5)
max_inventory = st.sidebar.number_input("Maximum Inventory per Product", min_value=0, value=100)
initial_inventory = st.sidebar.number_input("Initial Inventory", min_value=0, value=0)
final_inventory = st.sidebar.number_input("Final Inventory", min_value=0, value=50)
working_hours_per_day = st.sidebar.number_input("Working hours per day", min_value=0, value=16)
working_days_per_month = st.sidebar.number_input("Working days per month", min_value=0, value=24)
total_hours_per_machine = working_hours_per_day*working_days_per_month
updated_parameters = {
    "Working Hours per Day": working_hours_per_day,
    "Working Days per Month": working_days_per_month,
    "Total Hours per Machine": total_hours_per_machine,
    "Max Inventory per Product": max_inventory,
    "Initial Inventory": initial_inventory,
    "Final Inventory": final_inventory,
    "Storage Cost (£)": storage_cost
}
col1, col2 = st.columns(2)
with col1:
    st.subheader("Market Demand:")
    market_demand_df = pd.DataFrame(market_demand).T
    market_demand_df.columns = [f"Prod {i}" for i in products]
    market_demand_df.index = [monthname[t + 1] for t in months]
    st.dataframe(market_demand_df)
with col2:
    st.subheader("Key Parameters:")
    parameters_df = pd.DataFrame(list(updated_parameters.items()), columns=["Key Parameters", "Value"])
    parameters_df.index = range(1, len(parameters_df) + 1)
    st.dataframe(parameters_df)

col1, col2 ,col3= st.columns(3)
with col1:
    st.subheader("Machine Availability (Hours per Month):")
    machine_avail_df = pd.DataFrame(machine_availability)
    machine_avail_df.index = [monthname[t + 1] for t in range(len(machine_availability["GR"]))]
    st.dataframe(machine_avail_df)
with col2:
    st.subheader("Processing Time (Hours per Unit):")
    processing_time_df = pd.DataFrame(processing_time)
    processing_time_df.index = [f"Prod {i}" for i in products]
    st.dataframe(processing_time_df)
with col3:
    machine_names_for_display = {abbreviation: machine_names.get(abbreviation, abbreviation) for abbreviation in machine_types.keys()}
    machine_types_df = pd.DataFrame(list(machine_types.values()), index=[machine_names_for_display.get(key, key) for key in machine_types.keys()], columns=["Units"])
    st.subheader("Each Machine Type Availability:")
    machine_types_df.index.name = "Machine Name"
    st.dataframe(machine_types_df)
st.markdown("""---""")


# Create Gurobi model
model = Model("Factory_Production_Optimization")

# Decision variables
MPROD = model.addVars(products, months, lb=0, name="MPROD")  # Manufactured quantities
SPROD = model.addVars(products, months, lb=0, name="SPROD")  # Sold quantities
HPROD = model.addVars(products, months, lb=0, name="HPROD")  # Held quantities

# Binary maintenance decision variables
MDown = model.addVars(machine_types.keys(), range(1, max(machine_types.values())+1), months[1:], vtype=GRB.BINARY, name="MDown")

# Objective function: Maximize total profit
model.setObjective(
    quicksum(profit[i] * SPROD[i, t] for i in products for t in months) -
    0.5 * quicksum(HPROD[i, t] for i in products for t in months),
    GRB.MAXIMIZE
)

# Machine capacity constraints
for machine, count in machine_types.items():
    for t in months[1:]:
        available_capacity = sum(machine_availability[machine][t-1] * total_hours_per_machine for _ in range(count))
        model.addConstr(
            quicksum(processing_time.get(machine, {}).get(i, 0) * MPROD[i, t] for i in products)
            <= available_capacity - quicksum(MDown[machine, n, t] * machine_availability[machine][t-1] * total_hours_per_machine for n in range(1, count + 1)),
            f"{machine}_capacity_month_{t}"
        )


# Maintenance scheduling constraints
for machine, count in machine_types.items():
    if machine == "GR":
        model.addConstr(quicksum(MDown[machine, n, t] for n in range(1, count + 1) for t in months[1:]) == 2, f"Two_maintenance_{machine}")
    else:
        for n in range(1, count + 1):
            model.addConstr(quicksum(MDown[machine, n, t] for t in months[1:]) == 1, f"One_maintenance_{machine}_{n}")

# Market demand constraints
for i in products:
    for t in months:
        model.addConstr(SPROD[i, t] <= market_demand[i][t], f"Market_demand_{i}_month_{t}")

# Sales limit constraints
for i in products:
    for t in months:
        model.addConstr(SPROD[i, t] <= MPROD[i, t] + HPROD[i, t], f"Sales_limit_{i}_month_{t}")

# Inventory balance constraints
for i in products:
    for t in range(1, 7):
        model.addConstr(HPROD[i, t-1] + MPROD[i, t] - SPROD[i, t] - HPROD[i, t] == 0, f"Stock_balance_{i}_month_{t}")

# Ensure final inventory is 50 at month 6
for i in products:
    model.addConstr(HPROD[i, 6] == final_inventory, f"End_inventory_{i}")

# Initial inventory constraints
for i in products:
    model.addConstr(HPROD[i, 1] == initial_inventory, f"Initial_inventory_{i}")

# Maximum holding inventory constraint
for i in products:
    for t in months:
        model.addConstr(HPROD[i, t] <= max_inventory, f"Max_hold_{i}_month_{t}")



st.markdown("""
    <style>
        .container-c2 p {
            font-size: 20px;
            margin-bottom: 20px;
        }
    </style>
    <div class="container-c2">
        <br><p>Click on the button below to solve the optimization problem.</p>
    </div>
""", unsafe_allow_html=True)
## Solve the model
if st.button("Solve Optimization"):
    model.optimize()
    if model.status == GRB.OPTIMAL:
        st.markdown("---")
        total_profit = model.objVal
        st.markdown(f"<h3>Total Profit : <span style='color:rgba(255, 75, 75, 1) ;'> <b>£{total_profit:.2f}</b></span></h3>",unsafe_allow_html=True)
        st.markdown("---")

        # Collect maintenance schedule data
        maintenance_records = []
        for t in months[1:]:
            for machine, count in machine_types.items():
                for n in range(1, count + 1):
                    if MDown[machine, n, t].x > 0.5:  # If machine is under maintenance
                        maintenance_records.append([t, machine, n])
        maintenance_df = pd.DataFrame(maintenance_records, columns=["Month", "Machine Type", "Machine Number"])


        # Prepare data for multi-bar clustered graph
        st.markdown(f"<h1>Machine Maintenance Schedule:</h1>", unsafe_allow_html=True)
        maintenance_counts = {t: {m: 0 for m in machine_types.keys()} for t in months[1:]}

        for _, row in maintenance_df.iterrows():
            maintenance_counts[row["Month"]][row["Machine Type"]] += 1

        maintenance_plot_df = pd.DataFrame(maintenance_counts).T  # Transpose for correct structure

        if not maintenance_df.empty:  # Only show the chart if maintenance exists
            fig, ax = plt.subplots(figsize=(10, 6))

            bar_width = 0.25  # Width of each bar
            x_indexes = np.arange(len(maintenance_plot_df))  # X positions for months

            # Plot each machine type as a separate bar in the clustered format
            for i, machine in enumerate(machine_types.keys()):
                ax.bar(x_indexes + i * bar_width, maintenance_plot_df[machine], width=bar_width, label=machine)

            # Formatting
            ax.set_title("Machine Maintenance Schedule", fontsize=14)
            ax.set_xlabel("Month", fontsize=12)
            ax.set_ylabel("Number of Machines in Maintenance", fontsize=12)
            ax.set_xticks(x_indexes + (len(machine_types.keys()) - 1) * bar_width / 2)  # Centering x-ticks
            ax.set_xticklabels([monthname[t] for t in months[1:]])
            ax.legend(title="Machine Type",loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True, alpha=0.7)

            # Show plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("✅ No maintenance required this period.")
        # Collect data for each product across months
        produced_data = {}
        sold_data = {}
        held_data = {}

        for product in products:
            produced_data[product] = [MPROD[product, t].x for t in months]
            sold_data[product] = [SPROD[product, t].x for t in months]
            held_data[product] = [HPROD[product, t].x for t in months]

        # Display Production Trends
        st.markdown(f"<h1>Production Trends:</h1>", unsafe_allow_html=True)
        fig_produced, ax_produced = plt.subplots(figsize=(8, 5))
        for product in products:
            ax_produced.plot(months, produced_data[product], label=f"Product {product}", marker='o', linestyle='-',
                             markersize=6)
        ax_produced.set_title("Production Over Time", fontsize=16)
        ax_produced.set_xlabel("Months", fontsize=12)
        ax_produced.set_ylabel("Quantity", fontsize=12)
        ax_produced.legend(title="Product", fontsize=10, loc='upper left',
                           bbox_to_anchor=(1, 1))  # Legend moved to right
        ax_produced.grid(True)
        st.pyplot(fig_produced)

        # Display Sales Trends
        st.markdown(f"<h1>Sales Trends:</h1>", unsafe_allow_html=True)
        fig_sold, ax_sold = plt.subplots(figsize=(8, 5))
        for product in products:
            ax_sold.plot(months, sold_data[product], label=f"Product {product}", marker='o', linestyle='-',
                         markersize=6)
        ax_sold.set_title("Sales Over Time", fontsize=16)
        ax_sold.set_xlabel("Months", fontsize=12)
        ax_sold.set_ylabel("Quantity", fontsize=12)
        ax_sold.legend(title="Product", fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))  # Legend moved to right
        ax_sold.grid(True)
        st.pyplot(fig_sold)

        # Display Storage Trends
        st.markdown(f"<h1>Inventory Trends:</h1>", unsafe_allow_html=True)
        fig_held, ax_held = plt.subplots(figsize=(8, 5))
        for product in products:
            ax_held.plot(months, held_data[product], label=f"Product {product}", marker='o', linestyle='-',
                         markersize=6)
        ax_held.set_title("Inventory Held Over Time", fontsize=16)
        ax_held.set_xlabel("Months", fontsize=12)
        ax_held.set_ylabel("Quantity", fontsize=12)
        ax_held.legend(title="Product", fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))  # Legend moved to right
        ax_held.grid(True)
        st.pyplot(fig_held)

        st.markdown("""---""")

        # Display Bar Graphs for Each Month
        for t in months:
            if t>0:
                st.subheader(f"**Month :** {monthname[t]}")
                col1, col2 = st.columns(2)
                with col1:
                    with col1:
                        fig, ax = plt.subplots(figsize=(4, 3))  # Compact plot size

                        # Data for the plot
                        monthly_data = {
                            "Manufactured": [MPROD[p, t].x for p in products],
                            "Sold": [SPROD[p, t].x for p in products],
                            "Held": [HPROD[p, t].x for p in products],
                        }
                        df = pd.DataFrame(monthly_data, index=[f"PROD {p}" for p in products])

                        # Plot the bar chart
                        df.plot(kind='bar', ax=ax, width=0.7)

                        # Adjust font sizes for title, axes labels, and ticks
                        ax.set_title(f"Production Overview - {monthname[t]}", fontsize=8)
                        ax.set_xlabel("Products", fontsize=7)
                        ax.set_ylabel("Quantity", fontsize=7)
                        ax.tick_params(axis='both', labelsize=6)

                        # Adjust legend
                        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))

                        # Optimize layout for compact space
                        plt.tight_layout(pad=1.0)

                        # Display the plot in Streamlit
                        st.pyplot(fig, use_container_width=False)

                with col2:
                    month_profit = (
                            sum(profit[p] * SPROD[p, t].x for p in products) -
                            storage_cost * sum(HPROD[p, t].x for p in products)
                    )
                    st.write(f"**Profit :** £{month_profit:.2f}")

                    # 🛠️ Display Machines Under Maintenance for the Current Month
                    maintenance_records_month = [
                        f"{machine_names_for_display[machine]} (Machine #{n})"
                        for machine, count in machine_types.items()
                        for n in range(1, count + 1)
                        if MDown[machine, n, t].x > 0.5
                        # Assuming MDown[machine, n, t].x is a condition for maintenance
                    ]

                    if maintenance_records_month:
                        st.markdown(f"Maintenance of : <span style='color:#1c6a90 ;'><b>{', '.join(maintenance_records_month)}</b></span>", unsafe_allow_html=True)
                    else:
                        st.write("✅ No machines were under maintenance this month.")

                    st.write(f"**Production Data for {monthname[t]}:**")

                    # Create a dictionary to store data per product
                    production_metrics = {
                        f"Product {p}": [
                            round(MPROD[p, t].x, 1),  # Manufactured this month
                            round(HPROD[p, t - 1].x if t > 1 else 0, 1),  # Previously Held (from last month)
                            round(SPROD[p, t].x, 1),  # Sold this month
                            round(HPROD[p, t].x, 1),  # Stored at end of month
                        ]
                        for p in products
                    }

                    # Convert dictionary to DataFrame
                    production_table = pd.DataFrame(
                        production_metrics,
                        index=["Manufactured", "Previously Held", "Sold", "Held"]
                    ).T
                    st.write(production_table)


                # Add a separator for clarity
                st.markdown("""---""")

        st.markdown("""
                    <style>
                        footer {
                            text-align: center;
                            background-color: #f1f1f1;
                            color: #333;
                            font-size: 19px;
                            margin-bottom:0px;
                        }
                        footer img {
                            width: 44px; /* Adjust size of the logo */
                            height: 44px;
                        }
                    </style>
                    <footer>
                        <h1>Author- Ashutosh <a href="https://www.linkedin.com/in/ashutoshpatel24x7/" target="_blank">
                        <img src="https://decisionopt.com/static/media/LinkedIn.a6ad49e25c9a6b06030ba1b949fcd1f4.svg" class="img" alt="Logo"/></h1>
                    </footer>
                """, unsafe_allow_html=True)
        st.markdown("""---""")

    else:
        st.error("No optimal solution found!")
        st.markdown("""
                    <style>
                        footer {
                            text-align: center;
                            background-color: #f1f1f1;
                            color: #333;
                            font-size: 19px;
                            margin-bottom:0px;
                        }
                        footer img {
                            width: 44px; /* Adjust size of the logo */
                            height: 44px;
                        }
                    </style>
                    <footer>
                        <h1>Author- Ashutosh <a href="https://www.linkedin.com/in/ashutoshpatel24x7/" target="_blank">
                        <img src="https://decisionopt.com/static/media/LinkedIn.a6ad49e25c9a6b06030ba1b949fcd1f4.svg" class="img" alt="Logo"/></h1>
                    </footer>
                """, unsafe_allow_html=True)
        st.markdown("""---""")
