import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.formula.api as smf

from hrpqct_database.dataclasses_hrpqct import HRpQCT_Dataset
from PIL import Image
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize

# pio.templates.default = "plotly_white+web-personal"
pio.templates.default = "plotly_white"

# Configure Kaleido engine for Docker environment
pio.kaleido.scope.default_width = 700
pio.kaleido.scope.default_height = 500

# pio.kaleido.scope.mathjax = None

# flake8: noqa: E501


class Statistics:
    """
    Statistics class for direct treatment and comparison of RedCap eCRS
    Author: Simone Poncioni, MSB (ARTORG Center for Biomedical Engineering Research)
    Date: September 2022
    """

    def __init__(self, df, df_patient, name, originator, showfig, savefig):
        """
        Initialization constructor of Statistics class
        Args:
            data (path): CSV full filepath to direct export from RedCap (Data labels) tab-delimited, header at 0th row
            name (str): name of the study
            ORIG (_type_): originator of the computed results (e.g. 'POS': acronym for POncioni Simone)
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        self.common_db = HRpQCT_Dataset(df=df)
        self.patient_db = HRpQCT_Dataset(df=df_patient)
        self.df = df
        self.patient_df = df_patient
        self.name = str(name)
        self.df["Study"] = name
        self.originator_s = str(originator)
        self.current_datetime = datetime.datetime.now()
        self.age_limit = int(37)
        self.color_g = "#66cc66"
        self.color_y = "#ffad3a"
        self.color_r = "#e2221d"
        self.showfig = showfig
        self.savefig = savefig
        self.outputdir = (
            Path(f"02_OUTPUT")
            / f"{self.patient_db.Study_name.values[0]}_{self.patient_df.UID.values[0]}"
        )
        self.outputdir.mkdir(parents=True, exist_ok=True)
        self.width = 600
        self.height = 600

    def print_head(self, rows):
        print(self.df.head(rows))
        return None

    def get_columns(self):
        print(list(self.df.columns))

    def get_dim(self):
        """get sample size and number of parameters in study (nb of columns)"""
        print(f"Sample size: {len(self.df)}")
        print(f"Parameters: {len(list(self.df.columns))}")
        return None

    def __linear_regression__(self, column1: str, column2: str, loglog: bool = False):
        """
        calculates linear regression of two columns
        Args: column1, column2 [dataframe columns]
        Returns: R^2, slope, p-value, interception, standard error [floats]
        """
        if loglog:
            col1 = np.array(np.log10(column1)).reshape(len(np.log10(column1)), 1)
            col2 = np.array(np.log10(column2)).reshape(len(np.log10(column2)), 1)
        else:
            col1 = np.array(column1).reshape(len(column1), 1)
            col2 = np.array(column2).reshape(len(column2), 1)

        columns = np.hstack((col1, col2))
        columns = columns[~np.isnan(columns).any(axis=1)]
        linreg = stats.linregress(columns[:, 0], columns[:, 1])

        r_squared = linreg.rvalue**2
        slope = linreg.slope
        p_value = linreg.pvalue
        intercept = linreg.intercept
        stderr = linreg.stderr
        return r_squared, slope, p_value, intercept, stderr

    def linear_regression_constraint(self, column1, column2, ref):
        """
        calculates linear regression of two columns with the contraint of going through a specific point
        Args:
            column1, column2: dataframe columns
            ref: y-value of specific point (x=37)
        Returns: slope and interception of the calculated linear regression
        """

        # define the objective function
        def objective(beta):
            return ((column2 - beta[0] - beta[1] * column1) ** 2).sum()

        # define the constraint function
        def constraint(beta):
            return beta[0] + beta[1] * self.age_limit - ref[0]

        # specify the constraint
        con = {"type": "eq", "fun": constraint}

        # set the initial guess for the coefficients
        beta0 = np.array([0, 0])

        # minimize the objective function subject to the constraint
        res = minimize(objective, beta0, constraints=con)

        # print the optimal coefficients
        intercept, slope = res.x
        return intercept, slope

    def stderr_regression_above_37(self, column1, column2, ref):
        """
        calculates the standard error of the regression for a certain property when the Age in the dataset is above 37 years
        Args:
            column1, column2: dataframe columns
            ref: y-value of specific point (x=37)
        Returns: standard error of the regression
        """
        x = np.array(
            column1[self.patient_df.Gender.values[0] == self.common_db.Gender][
                column1 > 37
            ]
        )
        y = np.array(
            column2[self.patient_df.Gender.values[0] == self.common_db.Gender][
                column1 > 37
            ]
        )
        data = np.stack((x, y), axis=1)
        data = data[~np.isnan(data).any(axis=1), :]

        def objective(beta):
            return ((data[:, 1] - beta[0] - beta[1] * data[:, 0]) ** 2).sum()

        def constraint(beta):
            return beta[0] + beta[1] * 37 - ref[0]

        con = {"type": "eq", "fun": constraint}
        beta0 = np.array([0, 0])
        res = minimize(objective, beta0, constraints=con)

        intercept, slope = res.x
        y_pred = intercept + slope * data[:, 0]
        residuals = data[:, 1] - y_pred
        stderr = np.sqrt((residuals**2).sum() / (len(data) - 2))

        return stderr

    def median_regression_above_37(self, column1, column2):
        """
        Perform median (quantile) regression for Age > 37.
        """
        # Filter by gender and age
        gender = self.patient_df.Gender.values[0]
        age_mask = (self.common_db.Gender == gender) & (column1 > self.age_limit)

        x = np.array(column1[age_mask])
        y = np.array(column2[age_mask])

        # Remove NaNs
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_mask]
        y = y[valid_mask]

        # Create DataFrame for quantreg
        df_reg = pd.DataFrame({"x": x, "y": y})

        # Fit quantile regression
        model = smf.quantreg("y ~ x", df_reg)
        res = model.fit(q=0.5)

        return res

    def linear_regression_constraint_paper(self, column1, column2, ref):
        """
        calculates linear regression of two columns with the contraint of going through a specific point
        Args:
            column1, column2: dataframe columns
            ref: y-value of specific point (x=37)
        Returns: slope and intercept of the calculated linear regression
        """
        x = np.array(
            column1[self.patient_df.Gender.values[0] == self.common_db.Gender][
                column1 >= self.age_limit
            ]
        )
        y = np.array(
            column2[self.patient_df.Gender.values[0] == self.common_db.Gender][
                column1 >= self.age_limit
            ]
        )
        data = np.stack((x, y), axis=1)
        data = data[~np.isnan(data).any(axis=1), :]

        # define the objective function
        def objective(beta):
            return ((data[:, 1] - beta[0] - beta[1] * data[:, 0]) ** 2).sum()

        # define the constraint function
        def constraint(beta):
            return beta[0] + beta[1] * self.age_limit - ref[0]

        # specify the constraint
        con = {"type": "eq", "fun": constraint}

        # set the initial guess for the coefficients
        beta0 = np.array([0, 0])

        # minimize the objective function subject to the constraint
        res = minimize(objective, beta0, constraints=con)

        # print the optimal coefficients
        intercept, slope = res.x
        return intercept, slope

    def control_factor(
        self, column1, column2, gender_needed, gender_investigated, reference
    ):
        """
        Calculates the "control factor" (stderr/change), needs to calc linear regression first,
        and then the standard error,
        separate for male and female
        Args:
        column1, column2: (linReg),
        gender_needed: column of dataset with the genders,
        gender_investigated: gender of the investigated group
        Returns:
        control factor, % change, stderr [float, float, float]
        """
        if gender_investigated == "Female":
            x = np.array(column1[gender_needed == "Female"][column1 > 37])
            y = np.array(column2[gender_needed == "Female"][column1 > 37])
            data = np.stack((x, y), axis=1)
            data = data[~np.isnan(data).any(axis=1), :]

        if gender_investigated == "Male":
            x = np.array(column1[gender_needed == "Male"][column1 > 37])
            y = np.array(column2[gender_needed == "Male"][column1 > 37])
            data = np.stack((x, y), axis=1)
            data = data[~np.isnan(data).any(axis=1), :]

        intercept, slope = self.linear_regression_constraint(
            data[:, 0], data[:, 1], reference
        )

        # calculate the residual sum of squares
        rss = ((data[:, 1] - intercept - slope * data[:, 0]) ** 2).sum()

        # calculate the degrees of freedom
        df = len(data[:, 1]) - 2

        # calculate the residual standard error
        rse = np.sqrt(rss / df)

        # add a constant vector of ones to x
        X = np.column_stack((np.ones(len(data[:, 0])), data[:, 0]))

        # calculate the covariance matrix of the coefficients
        cov = rse**2 * np.linalg.inv(np.dot(X.T, X))

        # calculate the standard errors of the coefficients
        stderr_intercept, stderr_slope = np.sqrt(np.diag(cov))

        cf = abs(slope) / rse
        # print(column2.name + " " + gender_investigated + ": " + str(cf))
        # print("change = " + str(slope))
        # print("stderr = " + str(rse))

        return cf, slope, rse

    def t_score(self, value, average, stdev):
        """
        calculates t-score (value - mean / stdev)
        Args:
            value: value of patient,
            reference: mean and stdev of reference group
        Returns:
            (float)
        """
        return (value - average) / stdev

    def save_figure(self, fig, column1, column2, typename):
        try:
            column_1_str = (
                column1.name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace(":", "")
                .replace("/", "_")
                .replace("²", "2")
            )
            column_2_str = (
                column2.name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace(":", "")
                .replace("/", "_")
                .replace("²", "2")
            )
            fig_name = f"{column_1_str}-{column_2_str}"
            typename_str = str(typename)
            if typename_str.endswith(".png"):
                fig.write_image(
                    str(self.outputdir / f"{fig_name}{typename}"), scale=5, format="png"
                )
            else:
                fig.write_image(
                    str(self.outputdir / f"{fig_name}{typename}"), scale=5, format="pdf"
                )
        except AttributeError:
            typename_str = str(typename)
            if typename_str.endswith(".png"):
                fig.write_image(typename_str, format="png")
            else:
                fig.write_image(typename_str, format="pdf")

    def get_radar_chart(self, patient_data, patient_UID):
        t_0 = [-1] * 7
        t_m1 = [-2.5] * 7
        t_pos = [10] * 7

        patient_values = [patient_data[i]._values[0] for i in range(len(patient_data))]
        patient_values.append(patient_values[0])

        # repeat first value to close the hexagon
        theta = [
            patient_data[0].name.split("[")[0].split(": ")[1:],
            patient_data[1].name.split("[")[0].split(": ")[1:],
            patient_data[2].name.split("[")[0].split(": ")[1:],
            patient_data[3].name.split("[")[0].split(": ")[1:],
            patient_data[4].name.split("[")[0].split(": ")[1:],
            patient_data[5].name.split("[")[0].split(": ")[1:],
            patient_data[0].name.split("[")[0].split(": ")[1:],
        ]

        # if in theta, one value contains y.Stress, replace with html
        theta = [
            " ".join(name).replace(
                "y.Stress",
                "<span style='font-size:30px'><sup>app</sup>σ<sub>y</sub></span>",
            )
            for name in theta
        ]

        # range of t-score
        radial_range_default = [-6, 0]
        if np.min(patient_values) > -6 and np.max(patient_values) < 0:
            radial_range = radial_range_default
        elif np.min(patient_values) < -6 and np.max(patient_values) < 0:
            radial_range = [(np.min(patient_values) - 1), radial_range_default[1]]
        elif np.min(patient_values) > -6 and np.max(patient_values) > 0:
            radial_range = [(radial_range_default[0]), (np.max(patient_values) + 1)]
        else:
            radial_range = radial_range_default

        color_g = "rgba(102, 204, 102, 0.7)"
        color_y = "rgba(255, 173, 58, 0.7)"
        color_r = "rgba(226, 34, 29, 0.7)"

        color_g_line = "rgba(102, 204, 102, 0.0)"
        color_y_line = "rgba(255, 173, 58, 0.0)"
        color_r_line = "rgba(226, 34, 29, 0.0)"

        color_p = "rgb(24,78,119)"
        lw = 5
        opacity_s = 1.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=t_pos,
                theta=theta,
                textposition="top center",
                name="T-Score",
                line_color=color_g_line,
                opacity=opacity_s,
                line_width=lw,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=t_0,
                theta=theta,
                fill="toself",
                name="T-Score",
                fillcolor=color_y,
                opacity=opacity_s,
                line_color=color_y_line,
                line_width=lw,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=t_m1,
                theta=theta,
                fill="toself",
                name="T-Score",
                fillcolor=color_r,
                opacity=opacity_s,
                line_color=color_r_line,
                line_width=lw,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=patient_values,
                theta=theta,
                line_color=color_p,
                fillcolor=color_p,
                line_width=lw,
                name=f"Patient {patient_UID}",
                showlegend=False,
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=radial_range,
                    tickangle=270,
                ),
                angularaxis=dict(tickfont=dict(size=20)),
            ),
            width=self.width,
            height=self.width,
        )
        fig.update_layout(polar={"bgcolor": color_g})
        fig.update_layout(
            font=dict(
                family="STIXGeneral, serif",
                size=20,
                color="black",
            ),
        )
        fig.update_traces(mode="lines", selector=dict(type="scatterpolar"))
        fig.update_layout(margin=dict(l=150, r=150, t=50, b=50), margin_autoexpand=True)
        fig.update_layout(
            {
                # "plot_bgcolor": "rgba(0, 0, 0, 0)",
                # "paper_bgcolor": "rgba(0, 0, 0, 0)",
                "plot_bgcolor": "rgba(255, 255, 255, 1)",
                "paper_bgcolor": "rgba(255, 255, 255, 1)",
            }
        )
        fig.update_traces()

        if self.savefig:
            self.save_figure(
                fig,
                None,
                None,
                self.outputdir
                / f"{patient_UID}_{patient_data[0].name.split(': ')[0]}_radar.png",
            )
        if self.showfig:
            fig.show()

    def get_radar_chart_paper(self, data_37, data_50, data_63, data_76, data_89, site):
        t_0 = [-1] * 7
        t_m1 = [-2.5] * 7
        t_pos = [10] * 7

        data_37.append(data_37[0])
        data_50.append(data_50[0])
        data_63.append(data_63[0])
        data_76.append(data_76[0])
        data_89.append(data_89[0])
        # repeat first value to close the hexagon
        theta = [
            "Tot.vBMD",
            "Ct.vBMD",
            "Tb.BV/TV",
            "Rel.Ct.Th",
            "Tb.DA",
            "<span style='font-size:30px'><sup>app</sup>σ<sub>y</sub></span>",
            "Tot.vBMD",
        ]

        # color_g = "#66cc66"
        # color_y = "#ffad3a"
        # color_r = "#e2221d"
        color_g = "rgba(102, 204, 102, 0.7)"
        color_y = "rgba(255, 173, 58, 0.7)"
        color_r = "rgba(226, 34, 29, 0.7)"

        color_g_line = "rgba(102, 204, 102, 0.0)"
        color_y_line = "rgba(255, 173, 58, 0.0)"
        color_r_line = "rgba(226, 34, 29, 0.0)"

        # color_g = "#69995D"
        # color_y = "#FFAE03"
        # color_r = "#d83232"

        # color_g = "#00B38A"
        # color_y = "#F2AC42"
        # color_r = "#EA324C"
        color_a = "rgb(24,78,119)"
        color_b = "rgb(30,96,145)"
        color_c = "rgb(26,117,159)"
        color_c2 = "rgb(22,138,173)"
        color_c3 = "rgb(52,160,164)"
        opacity_s = 1.0
        lw = 5

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=t_pos,
                theta=theta,
                textposition="top center",
                name="T-Score",
                line_color=color_g_line,
                opacity=opacity_s,
                line_width=lw,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=t_0,
                theta=theta,
                fill="toself",
                name="T-Score",
                fillcolor=color_y,
                opacity=opacity_s,
                line_color=color_y_line,
                line_width=lw,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=t_m1,
                theta=theta,
                fill="toself",
                name="T-Score",
                fillcolor=color_r,
                opacity=opacity_s,
                line_color=color_r_line,
                line_width=lw,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=data_37,
                theta=theta,
                line_color=color_a,
                fillcolor=color_a,
                line_width=lw,
                # name="37",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=data_50,
                theta=theta,
                line_color=color_b,
                fillcolor=color_b,
                line_width=lw,
                # name="50",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=data_63,
                theta=theta,
                line_color=color_c,
                fillcolor=color_c,
                line_width=lw,
                # name="63",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=data_76,
                theta=theta,
                line_color=color_c2,
                fillcolor=color_c2,
                line_width=lw,
                # name="76",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=data_89,
                theta=theta,
                line_color=color_c3,
                fillcolor=color_c3,
                line_width=lw,
                # name="89",
                showlegend=False,
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-6.5, 0],
                    tickangle=270,
                ),
                angularaxis=dict(
                    tickfont=dict(size=20)  # Set font size for all theta labels
                ),
            ),
            width=self.width,
            height=self.width,
        )
        fig.update_layout(polar={"bgcolor": color_g})
        fig.update_layout(
            # title={
            #     "text": f"<b>T-Score [SD]</b><br>{self.patient_df.Gender.values[0]} {site}",
            #     "y": 0.95,
            #     "x": 0.5,
            #     "xanchor": "center",
            #     "yanchor": "top",
            # },
            font=dict(
                # family="unified sans, sans-serif",
                family="STIXGeneral, serif",
                size=20,
                color="black",
            ),
        )

        # fig.update_layout(
        #     legend=dict(
        #         yanchor="top",
        #         y=1.2,
        #         xanchor="left",
        #         x=-0.2,
        #         font=dict(size=18),
        #         bgcolor="rgba(255,255,255,0.0)",  # 0.0 = transparent
        #         title="<b>Age (years)</b>",
        #     )
        # )

        # fig.update_layout(legend=dict(yanchor="top", y=0.96, xanchor="left", x=1))
        fig.update_traces(mode="lines", selector=dict(type="scatterpolar"))
        fig.update_layout(margin=dict(l=150, r=150, t=50, b=50), margin_autoexpand=True)
        fig.update_layout(
            {
                "plot_bgcolor": "rgba(0, 0, 0, 0)",
                "paper_bgcolor": "rgba(0, 0, 0, 0)",
            }
        )
        fig.update_traces()

        if self.savefig:
            self.save_figure(
                fig,
                None,
                None,
                f"radar_paper_{self.patient_df.Gender.values[0]}_{site}.pdf",
            )
        if self.showfig:
            fig.show()

        return None

    def get_hist(self, column):
        """
        Get histogram of a specified column
        Args:
            column (str): string name of column contained in dataframe to be plotted
        """
        fig = px.histogram(self.df, x=column, nbins=20, title=self.name)
        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            # xaxis_title=column.name,
        )
        fig.show()
        return None

    def get_scatterplot(
        # self, column1, column2, color_s, linreg=False, savefig=False
        self,
        column1,
        column2,
        color_s,
        symbol_s,
        linreg=False,
        savefig=False,
    ):
        """
        Get scatterplot of 2 specified columns

        Args:
            column1 (pd.Series): first column to plot
            column2 (pd.Series): second column to plot
            color_s (pd.Series): third column to categorize the scatterplot
            symbol_s (pd.Series): fourth column to distinguishe de markers in the plot
            linreg: boolean if linreg should be showed
            savefig: boolean if figure should be saved
        """

        r_squared, _, p_value, _, _ = self.__linear_regression__(column1, column2)
        if linreg:
            # if p_value < 0.05:
            fig = px.scatter(
                x=column1,
                y=column2,
                color=color_s,
                symbol=symbol_s,
                trendline="ols",
                trendline_scope="overall",
                title=self.name,
                # log_x=True,
                # log_y=True
            )

            text_annotation = "R-squared  =<br>p-value >"
            res_annotation = f"{r_squared:.3f}<br>{0.01:.2f}"
            fig.add_annotation(
                x=1.22,
                y=0.2,
                align="right",
                text=text_annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
            )
            fig.add_annotation(
                x=1.29,
                y=0.2,
                align="right",
                text=res_annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
            )
            """
                fig = px.scatter(
                    x=column1 "Female",
                    y=column2 "Female",
                    color=color_s,
                    symbol=symbol_s,
                    trendline="ols",
                    # trendline_scope="overall",
                    title=self.name
                    # log_x=True,
                    # log_y=True
                )
                """
            """
                text_annotation = "R-squared  =<br>p-value  ="
                res_annotation = f"{r_squared:.5f}<br>{p_value:.5f}"
                fig.add_annotation(
                    x=1.15,
                    y=0.2,
                    align="right",
                    text=text_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
                fig.add_annotation(
                    x=1.3,
                    y=0.2,
                    align="right",
                    text=res_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
                """
        else:
            fig = px.scatter(
                x=column1,
                y=column2,
                color=color_s,
                symbol=symbol_s,
                title=self.name,
            )

        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            legend_title="Study",
        )

        if self.savefig:
            self.save_figure(fig, column1, column2, "scplot.png")
        if self.showfig:
            fig.show()
        return None

    def get_scatterplot_patient_assessment(
        self,
        patient_prop01,
        patient_prop02,
        column1,
        column2,
        x,
        slope,
        intercept,
        reference,
    ):
        intercept_plus_1std = (reference[0] + reference[1]) - slope * self.age_limit
        intercept_minus_1std = (reference[0] - reference[1]) - slope * self.age_limit
        intercept_plus_2std = (
            reference[0] + (2 * reference[1])
        ) - slope * self.age_limit
        intercept_minus_2d5std = (
            reference[0] - (2.5 * reference[1])
        ) - slope * self.age_limit
        intercept_minus_3d5std = (
            reference[0] - (3.5 * reference[1])
        ) - slope * self.age_limit

        intercept_ranges = [
            intercept_plus_2std,
            intercept_plus_1std,
            intercept,
            intercept_minus_1std,
            intercept_minus_2d5std,
            intercept_minus_3d5std,
            intercept_minus_3d5std,
        ]
        reference_list = [
            reference[0] + (2 * reference[1]),
            reference[0] + reference[1],
            reference[0],
            reference[0] - reference[1],
            reference[0] - (2.5 * reference[1]),
            reference[0] - (3.5 * reference[1]),
            reference[0] - (3.5 * reference[1]),
        ]

        colors = [
            self.color_g,
            self.color_g,
            self.color_g,
            self.color_g,
            self.color_y,
            self.color_r,
            self.color_r,
        ]
        t_score_label = [
            "+ 2",
            "+ 1",
            "  0",
            "\u2013 1",
            "\u2013 2.5",
            "\u2013 3.5",
            "\u2013 3.5",
        ]

        # encode to utf-8
        t_score_label = [x.encode("utf-8") for x in t_score_label]

        fig = px.scatter(
            data_frame=self.common_db.df,
            x=column1,
            y=column2,
            title=self.name,
            symbol=self.common_db.df["Gender"],
            color=self.common_db.df["Study name"],
        )

        for i, intercept_mod in enumerate(intercept_ranges):
            if i < 6:
                fill_s = "tonexty"
            else:
                fill_s = "none"

            # curve mean >= age limit
            x_lin = np.arange(start=np.min(x), stop=np.max(x) + 1, step=1)
            y_s = slope * x_lin + intercept_mod
            fig.add_trace(
                go.Scatter(
                    x=x_lin,
                    y=y_s,
                    mode="lines",
                    fill=fill_s,
                    marker_color=colors[i],
                    opacity=1,
                    name=f"{self.patient_df.Gender.values[0]} T-Score = {t_score_label[i].decode('utf-8')} (SD)",
                    # showlegend=showlegend_s,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        for i, intercept_mod in enumerate(intercept_ranges):
            if i >= 1 and i < 6:
                fill_s = "tonexty"
            else:
                fill_s = None

            # curve mean =< age limit
            x_lin = np.arange(start=0, stop=self.age_limit + 1, step=1)
            fig.add_trace(
                go.Scatter(
                    x=x_lin,
                    y=[reference_list[i]] * len(x_lin),
                    mode="lines",
                    fill=fill_s,
                    marker_color=colors[i],
                    name=f"{self.patient_df.Gender.values[0]} average",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=[patient_prop01.values[0]],
                y=[patient_prop02.values[0]],
                mode="markers",
                marker=dict(
                    size=20,
                    symbol="hexagram",
                    line_color="#2F4F4F",
                    color="#DC3912",
                    line_width=2,
                ),
                name="Patient",
            )
        )

        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            xaxis_range=[
                min(self.common_db.Age),
                max(self.common_db.Age) + 5,
            ],
            # yaxis_range=[0, reference[0] + 3 * reference[1]],
            yaxis_range=[0, np.max(column2) + 0.2 * np.max(column2)],
            legend_title="Study",
            legend_font_size=12,
        )

        fig.update_layout(
            title={
                "text": "<b>Patient assessment against reference values [SD]</b>",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(
                # family="Montserrat, monospace",
                family="unified sans, sans-serif",
                size=16,
                color="black",
            ),
        )

        if self.savefig:
            self.save_figure(
                fig,
                column1,
                column2,
                typename="_scatterplot_patient_assessment.png",
            )
        if self.showfig:
            fig.show()

    def get_scatterplot_patient_assessment_paper(
        self,
        column1,
        column2,
        x,
        slope,
        intercept,
        reference,
    ):
        intercept_plus_1std = (reference[0] + reference[1]) - slope * self.age_limit
        intercept_minus_1std = (reference[0] - reference[1]) - slope * self.age_limit
        intercept_plus_2std = (
            reference[0] + (2 * reference[1])
        ) - slope * self.age_limit
        intercept_minus_2d5std = (
            reference[0] - (2.5 * reference[1])
        ) - slope * self.age_limit
        intercept_minus_3d5std = (
            reference[0] - (3.5 * reference[1])
        ) - slope * self.age_limit

        intercept_ranges = [
            intercept_plus_2std,
            intercept_plus_1std,
            intercept,
            intercept_minus_1std,
            intercept_minus_2d5std,
            intercept_minus_3d5std,
            intercept_minus_3d5std,
        ]
        reference_list = [
            reference[0] + (2 * reference[1]),
            reference[0] + reference[1],
            reference[0],
            reference[0] - reference[1],
            reference[0] - (2.5 * reference[1]),
            reference[0] - (3.5 * reference[1]),
            reference[0] - (3.5 * reference[1]),
        ]

        colors = [
            self.color_g,
            self.color_g,
            self.color_g,
            self.color_g,
            self.color_y,
            self.color_r,
            self.color_r,
        ]
        t_score_label = [
            "+ 2",
            "+ 1",
            "   0",
            "\u2013 1",
            "\u2013 2.5",
            "\u2013 3.5",
            "\u2013 3.5",
        ]

        # encode to utf-8
        t_score_label = [x.encode("utf-8") for x in t_score_label]

        # Create scatter plot
        fig = px.scatter(
            data_frame=self.common_db.df,
            x=column1,
            y=column2,
            title=self.name,
            color=self.common_db.df["Gender"],
            color_discrete_sequence=["#008080", "#9D44B5"],
        )

        # Update traces for each gender
        for trace in fig.data:
            if trace.name == "Male":
                trace.update(
                    marker=dict(
                        color="#008080",
                        opacity=0.6,
                        size=10,
                        line=dict(color="white", width=1),
                    )
                )
            elif trace.name == "Female":
                trace.update(
                    marker=dict(
                        color="#9D44B5",
                        opacity=0.6,
                        size=10,
                        line=dict(color="white", width=1),
                    )
                )

        for i, intercept_mod in enumerate(intercept_ranges):
            if i < 6:
                fill_s = "tonexty"
            else:
                fill_s = "none"

            # curve mean >= age limit
            x_lin = np.arange(start=np.min(x), stop=np.max(x) + 1, step=1)
            y_s = slope * x_lin + intercept_mod

            # Add the actual trace
            fig.add_trace(
                go.Scatter(
                    x=x_lin,
                    y=y_s,
                    mode="lines",
                    fill=fill_s,
                    marker_color=colors[i],
                    opacity=1,
                    name=f"{t_score_label[i].decode('utf-8')}",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Add text annotation to the right of the line
            fig.add_annotation(
                x=x_lin[-1] + 1,  # Last x value
                y=y_s[-1],  # Corresponding y value
                text=f"{t_score_label[i].decode('utf-8')} SD",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(color=colors[i]),
            )

        for i, intercept_mod in enumerate(intercept_ranges):
            if i >= 1 and i < 6:
                fill_s = "tonexty"
            else:
                fill_s = None

            # curve mean =< age limit
            x_lin = np.arange(start=0, stop=self.age_limit + 1, step=1)
            fig.add_trace(
                go.Scatter(
                    x=x_lin,
                    y=[reference_list[i]] * len(x_lin),
                    mode="lines",
                    fill=fill_s,
                    marker_color=colors[i],
                    name=f"{self.patient_df.Gender.values[0]} average",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            xaxis_range=[
                min(self.common_db.Age),
                max(self.common_db.Age),
            ],
            yaxis_range=[0, np.max(column2) + 0.2 * np.max(column2)],
            legend_title="Sex",
            legend_font_size=12,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.9,
                xanchor="center",
                x=0.5,
                title_text="",
                font=dict(size=20),
                traceorder="normal",
            ),
            margin=dict(l=20, r=60, t=50, b=70),
        )

        fig.update_layout(
            title={
                "text": "<b>Normative Reference Values (SD)</b>",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(
                family="Outfit, Cabin, sans-serif",
                size=16,
                color="black",
            ),
        )
        # font=["Outfit", "Cabin"],

        if self.savefig:
            self.save_figure(
                fig,
                column1,
                column2,
                typename="_scatterplot_patient_assessment.pdf",
            )
        if self.showfig:
            fig.show()

    def get_scatterplot_patient_assessment_quadratic(
        self,
        column1,
        column2,
        age_lim,
        quadratic_coeffs,
        reference,
    ):
        intercept, linear_coeff, quadratic_coeff = quadratic_coeffs
        intercept_plus_1std = (reference[0] + reference[1]) - (
            linear_coeff * age_lim + quadratic_coeff * age_lim**2
        )
        intercept_minus_1std = (reference[0] - reference[1]) - (
            linear_coeff * age_lim + quadratic_coeff * age_lim**2
        )
        intercept_plus_2std = (reference[0] + (2 * reference[1])) - (
            linear_coeff * age_lim + quadratic_coeff * age_lim**2
        )
        intercept_minus_2d5std = (reference[0] - (2.5 * reference[1])) - (
            linear_coeff * age_lim + quadratic_coeff * age_lim**2
        )
        intercept_minus_3d5std = (reference[0] - (3.5 * reference[1])) - (
            linear_coeff * age_lim + quadratic_coeff * age_lim**2
        )

        intercept_ranges = [
            intercept_plus_2std,
            intercept_plus_1std,
            intercept,
            intercept_minus_1std,
            intercept_minus_2d5std,
            intercept_minus_3d5std,
            intercept_minus_3d5std,
        ]
        reference_list = [
            reference[0] + (2 * reference[1]),
            reference[0] + reference[1],
            reference[0],  # This is the mean reference
            reference[0] - reference[1],
            reference[0] - (2.5 * reference[1]),
            reference[0] - (3.5 * reference[1]),
            reference[0] - (3.5 * reference[1]),
        ]

        colors = [
            self.color_g,
            self.color_g,
            self.color_g,
            self.color_g,
            self.color_y,
            self.color_r,
            self.color_r,
        ]
        t_score_label = [
            "+ 2",
            "+ 1",
            "   0",
            "\u2013 1",
            "\u2013 2.5",
            "\u2013 3.5",
            "\u2013 3.5",
        ]

        # encode to utf-8
        t_score_label = [x.encode("utf-8") for x in t_score_label]

        # color female grey if patient sex is male, and viceversa
        color_discrete_sequence_s = ["#008080", "#9D44B5"]
        if self.patient_db.Gender.values[0] == "Male":
            df = self.common_db.df[self.common_db.df["Gender"] == "Male"]
        elif self.patient_db.Gender.values[0] == "Female":
            df = self.common_db.df[self.common_db.df["Gender"] == "Female"]

        # filter column1 and column2 with the indices kept in df
        column1 = column1[df.index]
        column2 = column2[df.index]
        # Create scatter plot
        fig = px.scatter(
            data_frame=df,
            x=column1,
            y=column2,
            title=self.name,
            color=df["Gender"],
            color_discrete_sequence=color_discrete_sequence_s,
        )

        # Update traces for each gender
        for trace in fig.data:
            if trace.name == "Male":
                trace.update(
                    marker=dict(
                        color=color_discrete_sequence_s[0],
                        size=10,
                        line=dict(color="white", width=1),
                    )
                )
            elif trace.name == "Female":
                trace.update(
                    marker=dict(
                        color=color_discrete_sequence_s[1],
                        size=10,
                        line=dict(color="white", width=1),
                    )
                )

        x_vals = np.linspace(
            age_lim,
            np.max(column1[self.patient_df.Gender.values[0] == self.common_db.Gender]),
            1000,
        )
        for i, (intercept, label) in enumerate(zip(intercept_ranges, t_score_label)):
            y_vals = intercept + linear_coeff * x_vals + quadratic_coeff * x_vals**2
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    fill="tonexty" if i < 6 else "none",
                    marker_color=colors[i],
                    name=f"{label.decode('utf-8')} SD",
                    showlegend=False,
                )
            )
            fig.add_annotation(
                x=x_vals[-1] + 1,
                y=y_vals[-1],
                text=f"{label.decode('utf-8')} SD",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(color=colors[i]),
            )

        # Add horizontal lines from age 18 to age_lim
        x_vals_horizontal = np.linspace(18, age_lim, 100)
        for i, (y_val, label) in enumerate(zip(reference_list, t_score_label)):
            fill_s = "tonexty" if 0 < i < 6 else "none"
            fig.add_trace(
                go.Scatter(
                    x=x_vals_horizontal,
                    y=[y_val] * len(x_vals_horizontal),
                    mode="lines",
                    fill=fill_s,
                    marker_color=colors[i],
                    name=f"{label.decode('utf-8')} SD",
                    showlegend=False,
                )
            )

        def calculate_grid_size(value_range):
            """
            Calculate the grid size based on the range of values.
            """
            magnitude = np.floor(np.log10(value_range))
            grid_size = 10**magnitude
            return grid_size

        def round_to_grid_limits(y_lim):
            """
            Round the y_lim to the nearest larger grid divisor.
            """
            y_min, y_max = y_lim
            value_range = y_max - y_min
            grid_size = calculate_grid_size(value_range)

            y_min_rounded = np.floor(y_min / grid_size) * grid_size
            y_max_rounded = np.ceil(y_max / grid_size) * grid_size

            return [y_min_rounded, y_max_rounded]

        # Original y_lim calculation
        y_lim_s = (
            [-0.01 * np.min(column2), 1 * np.max(column2)]
            if np.min(y_vals) < 0
            else [0, 1 * np.max(column2)]
        )

        # Round y_lim to the nearest larger grid divisor
        y_lim = round_to_grid_limits(y_lim_s)

        #! delete after plotting for paper Tot.vBMD!
        # y_lim = [0, 2.5]

        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            xaxis_range=[
                min(self.common_db.Age),
                100,
            ],
            yaxis_range=y_lim,
            legend_title="Sex",
            legend_font_size=12,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.025,
                xanchor="center",
                x=0.5,
                title_text="",
                font=dict(size=20),
                # font_color="darkgrey",
                traceorder="normal",
            ),
            margin=dict(l=20, r=50, t=110, b=70),
            xaxis=dict(
                mirror=True,
                ticks="outside",
                showline=True,
            ),
            yaxis=dict(
                mirror=True,
                ticks="outside",
                showline=True,
            ),
        )

        fig.update_layout(
            title={
                "text": "<b>Normative Reference Values (SD)</b>",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font=dict(
                family="Outfit, Cabin, sans-serif",
                size=16,
                color="black",
            ),
        )

        if self.savefig:
            self.save_figure(
                fig,
                column1,
                column2,
                typename="_scatterplot_patient_assessment.pdf",
            )
        if self.showfig:
            fig.show()

        return intercept_ranges, reference_list

    def quadratic_regression_constraint(self, prop1, prop2, age):
        """
        Calculates quadratic regression of two columns with the constraint of going through a specific point
        and having a slope of 0 at that point.
        Args:
            prop1, prop2: dataframe columns
            age: specific age for the constraints
        Returns: coefficients of the calculated quadratic regression
        """

        def prop_mean_below_age(prop1, prop2, age):
            # Get the mean value at the specified age
            mean_at_age = np.mean(prop2[prop1 <= age])
            return mean_at_age

        # Filter data to include only valid indices
        valid_indices = ~np.isnan(prop1) & ~np.isnan(prop2)
        prop1 = prop1[valid_indices]
        prop2 = prop2[valid_indices]

        mean_at_age = prop_mean_below_age(prop1, prop2, age)

        # Definition of quadratic function as per 'Age dependence of BMD' (PhZ, 2024)
        def objective(beta):
            return ((prop2 - beta[0] - beta[1] * prop1 - beta[2] * prop1**2) ** 2).sum()

        # Definition of constraint 1:
        # ---> The quadratic function should go through the mean value at the specified age
        def constraint_point(beta):
            return beta[0] + beta[1] * age + beta[2] * age**2 - mean_at_age

        # Definition of constraint 2:
        # ---> The slope of the quadratic function at the specified age should be 0
        def constraint_slope(beta):
            return beta[1] + 2 * beta[2] * age

        cons = [
            {"type": "eq", "fun": constraint_point},
            {"type": "eq", "fun": constraint_slope},
        ]

        # Initial guess for coeffs a, b, c (intercept, linear term, quadratic term)
        beta0 = np.array([0, 0, 0])

        res = minimize(objective, beta0, constraints=cons)
        intercept, linear_coeff, quadratic_coeff = res.x

        return intercept, linear_coeff, quadratic_coeff

    def patient_assessment(
        self,
        patient_prop01: list,
        patient_prop02: list,
        column1: list,
        column2: list,
        reference: list,
        PAPER: bool = False,
        QUADRATIC: bool = False,
    ) -> pd.DataFrame:
        """
        Patient assessment is a wrapper function that returns the results to be included in the patient report.

        Args:
            column1 (list): property 1 to be assessed (e.g. age)
            column2 (list): property 2 to be assessed (e.g. Tibia_cortical_thickness_mm)
            reference (list): reference values based on normative database (e.g. [mean, std])
        """

        x = np.array(
            column1[self.patient_df.Gender.values[0] == self.common_db.Gender][
                column1 >= self.age_limit
            ]
        )
        y = np.array(
            column2[self.patient_df.Gender.values[0] == self.common_db.Gender][
                column1 >= self.age_limit
            ]
        )
        data = np.stack((x, y), axis=1)
        data = data[~np.isnan(data).any(axis=1), :]

        if QUADRATIC:
            intercept, linear_coeff, quadratic_coeff = (
                self.quadratic_regression_constraint(
                    column1[self.patient_df.Gender.values[0] == self.common_db.Gender],
                    column2[self.patient_df.Gender.values[0] == self.common_db.Gender],
                    self.age_limit,
                )
            )
        else:
            intercept, slope = self.linear_regression_constraint(
                data[:, 0], data[:, 1], reference
            )

        if PAPER == True:
            if QUADRATIC:
                self.get_scatterplot_patient_assessment_quadratic(
                    column1,
                    column2,
                    self.age_limit,
                    quadratic_coeffs=(intercept, linear_coeff, quadratic_coeff),
                    reference=reference,
                )

                # evaluate f at x = age_limit
                y_age_lim = (
                    intercept
                    + linear_coeff * self.age_limit
                    + quadratic_coeff * self.age_limit**2
                )

                fd = 2 * quadratic_coeff * self.age_limit + linear_coeff
                assert np.isclose(
                    reference[0], y_age_lim, atol=np.finfo(float).eps
                ), f"Intercept not close to reference mean ({reference[0]}, {y_age_lim})"

                assert np.isclose(fd, 0, atol=1e-3), f"Slope not close to 0 ({fd})"

            else:
                self.get_scatterplot_patient_assessment_paper(
                    column1,
                    column2,
                    x=data[:, 0],
                    slope=slope,
                    intercept=intercept,
                    reference=reference,
                )
        else:
            self.get_scatterplot_patient_assessment(
                patient_prop01,
                patient_prop02,
                column1,
                column2,
                x=data[:, 0],
                slope=slope,
                intercept=intercept,
                reference=reference,
            )

        if QUADRATIC:
            results_table = pd.DataFrame(
                {
                    "Property 1": patient_prop01,
                    "Property 2": patient_prop02,
                    "$\\mu_{ref}$": reference[0],
                    "$\\sigma_{ref}$": reference[1],
                    "a": intercept,
                    "b": linear_coeff,
                    "c": quadratic_coeff,
                }
            )
        else:
            results_table = pd.DataFrame(
                {
                    "Property 1": patient_prop01,
                    "Property 2": patient_prop02,
                    "Reference mean": reference[0],
                    "Reference std": reference[1],
                    "Slope": slope,
                    "Intercept": intercept,
                }
            )

        return results_table

    def get_scatterplot_gender(
        self,
        column1,
        column2,
        color_s,
        symbol_s,
        linreg=False,
        loglog=False,
        savefig=False,
    ):
        """
        Get scatterplot of 2 specified columns, with linReg separated according to symbol_s in the log log space

        Args:
            column1 (pd.Series): first column to plot
            column2 (pd.Series): second column to plot
            color_s (pd.Series): third column to categorize the scatterplot
            symbol_s (pd.Series): fourth column to distinguishe the markers in the plot and create separate linregs
            linreg: boolean if linreg should be showed
            loglog: boolean if figure should be im the loglog space
            savefig: boolean if figure should be saved
        """

        columns = np.transpose(np.vstack((column1, column2)))
        index = ~np.isnan(columns).any(axis=1)

        if loglog:
            bool = True
        else:
            bool = False

        fig = px.scatter(
            x=columns[index][:, 0],
            y=columns[index][:, 1],
            color=color_s[index],
            symbol=symbol_s[index],
            title=self.name,
            log_x=bool,
            log_y=bool,
        )

        if linreg:
            column1_male = columns[index][:, 0][symbol_s[index] == "Male"]
            column2_male = columns[index][:, 1][symbol_s[index] == "Male"]
            column1_female = columns[index][:, 0][symbol_s[index] == "Female"]
            column2_female = columns[index][:, 1][symbol_s[index] == "Female"]

            (
                r_squared_male,
                slope_male,
                p_value_male,
                intercept_male,
                stderr_male,
            ) = self.__linear_regression__(column1_male, column2_male, loglog=bool)
            (
                r_squared_female,
                slope_female,
                p_value_female,
                intercept_female,
                stderr_female,
            ) = self.__linear_regression__(column1_female, column2_female, loglog=bool)

            if loglog:
                y_m = column1_male**slope_male * 10**intercept_male
                y_f = column1_female**slope_female * 10**intercept_female

            else:
                y_m = slope_male * column1_male + intercept_male
                y_f = slope_female * column1_female + intercept_female

            if p_value_male < 0.05:
                fig.add_trace(
                    go.Scatter(
                        x=column1_male,
                        y=y_m,
                        mode="lines",
                        marker_color="rgb(0, 0, 255)",
                        name="Male",
                    ),
                    row=1,
                    col=1,
                )
                text_annotation = "R-squared m =<br>p-value m <"
                res_annotation = f"{r_squared_male:.3f}<br>{0.01:.2f}"

                fig.add_annotation(
                    x=0.90,
                    y=0.15,
                    align="right",
                    text=text_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
                fig.add_annotation(
                    x=0.97,
                    y=0.15,
                    align="right",
                    text=res_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )

            if p_value_female < 0.05:
                fig.add_trace(
                    go.Scatter(
                        x=column1_female,
                        y=y_f,
                        mode="lines",
                        marker_color="rgb(255, 0, 0)",
                        name="Female",
                    ),
                    row=1,
                    col=1,
                )

                text_annotation = "R-squared f =<br>p-value f <"
                res_annotation = f"{r_squared_female:.3f}<br>{0.01:.2f}"

                fig.add_annotation(
                    x=0.90,
                    y=0.05,
                    align="right",
                    text=text_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
                fig.add_annotation(
                    x=0.97,
                    y=0.05,
                    align="right",
                    text=res_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )

        fig.update_layout(
            autosize=False,
            width=self.width,
            height=self.height,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            legend_title="Study",
        )
        if loglog:
            if savefig:
                self.save_figure(
                    fig, column1, column2, typename="_scatterplot_gender_loglog.png"
                )
        else:
            if savefig:
                self.save_figure(
                    fig, column1, column2, typename="_scatterplot_gender.png"
                )
        fig.show()
        return None

    def get_scatterplot_gender_AgeSeparated(
        self,
        column1,
        column2,
        color_s,
        symbol_s,
        linreg=False,
        loglog=False,
        savefig=False,
    ):
        """
        Get scatterplot of 2 specified columns, with linReg separated according to symbol_s and
        distinguished for <=37 and >37 in the loglog space
        Args:
            column1 (pd.Series): first column to plot
            column2 (pd.Series): second column to plot
            color_s (pd.Series): third column to categorize the scatterplot
            symbol_s (pd.Series): fourth column to distinguishe the markers in the plot and create separate linregs
            linreg: boolean if linreg should be showed
            loglog: boolean if figure should be im the loglog space
            savefig: boolean if figure should be saved
        """

        columns = np.transpose(np.vstack((column1, column2)))
        index = ~np.isnan(columns).any(axis=1)

        if loglog:
            bool = True
        else:
            bool = False

        fig = px.scatter(
            x=columns[index][:, 0],
            y=columns[index][:, 1],
            color=color_s[index],
            symbol=symbol_s[index],
            title=self.name,
            log_x=bool,
            log_y=bool,
        )

        if linreg:
            column1_male = columns[index][:, 0][symbol_s[index] == "Male"]
            column1_male_above37 = column1_male > 37
            column1_male_bellow37 = column1_male <= 37
            column2_male = columns[index][:, 1][symbol_s[index] == "Male"]
            column1_female = columns[index][:, 0][symbol_s[index] == "Female"]
            column1_female_above37 = column1_female > 37
            column1_female_bellow37 = column1_female <= 37
            column2_female = columns[index][:, 1][symbol_s[index] == "Female"]

            (
                r_squared_male_a37,
                slope_male_a37,
                p_value_male_a37,
                intercept_male_a37,
                stderr_male_a37,
            ) = self.__linear_regression__(
                column1_male[column1_male_above37],
                column2_male[column1_male_above37],
                loglog=bool,
            )
            (
                r_squared_male_b37,
                slope_male_b37,
                p_value_male_b37,
                intercept_male_b37,
                stderr_male_b37,
            ) = self.__linear_regression__(
                column1_male[column1_male_bellow37],
                column2_male[column1_male_bellow37],
                loglog=bool,
            )
            (
                r_squared_female_a37,
                slope_female_a37,
                p_value_female_a37,
                intercept_female_a37,
                stderr_female_a37,
            ) = self.__linear_regression__(
                column1_female[column1_female_above37],
                column2_female[column1_female_above37],
                loglog=bool,
            )
            (
                r_squared_female_b37,
                slope_female_b37,
                p_value_female_b37,
                intercept_female_b37,
                stderr_female_b37,
            ) = self.__linear_regression__(
                column1_female[column1_female_bellow37],
                column2_female[column1_female_bellow37],
                loglog=bool,
            )

            if loglog:
                y_m_above37 = (
                    column1_male[column1_male_above37] ** slope_male_a37
                    * 10**intercept_male_a37
                )
                y_m_bellow37 = (
                    column1_male[column1_male_bellow37] ** slope_male_b37
                    * 10**intercept_male_b37
                )
                y_f_above37 = (
                    column1_female[column1_female_above37] ** slope_female_a37
                    * 10**intercept_female_a37
                )
                y_f_bellow37 = (
                    column1_female[column1_female_bellow37] ** slope_female_b37
                    * 10**intercept_female_b37
                )

            else:
                y_m_above37 = (
                    slope_male_a37 * column1_male[column1_male_above37]
                    + intercept_male_a37
                )
                y_m_bellow37 = (
                    slope_male_b37 * column1_male[column1_male_bellow37]
                    + intercept_male_b37
                )
                y_f_above37 = (
                    slope_female_a37 * column1_female[column1_female_above37]
                    + intercept_female_a37
                )
                y_f_bellow37 = (
                    slope_female_b37 * column1_female[column1_female_bellow37]
                    + intercept_female_b37
                )

            if p_value_male_a37 <= 0.05:
                fig.add_trace(
                    go.Scatter(
                        x=column1_male[column1_male_above37],
                        y=y_m_above37,
                        mode="lines",
                        marker_color="rgb(0, 0, 255)",
                        name="Male >37",
                    ),
                    row=1,
                    col=1,
                )

                text_annotation = "R-squared >37 m =<br>p-value >37 m ="
                res_annotation = f"{r_squared_male_a37:.5f}<br>{p_value_male_a37:.5f}"

                fig.add_annotation(
                    x=1.25,
                    y=0.1,
                    align="right",
                    text=text_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
                fig.add_annotation(
                    x=1.35,
                    y=0.1,
                    align="right",
                    text=res_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )

                fig.add_trace(
                    go.Scatter(
                        x=column1_male[column1_male_bellow37],
                        y=y_m_bellow37,
                        mode="lines",
                        marker_color="rgb(153, 153, 255)",
                        name="Male <=37",
                    ),
                    row=1,
                    col=1,
                )

            if p_value_female_a37 <= 0.05:
                fig.add_trace(
                    go.Scatter(
                        x=column1_female[column1_female_above37],
                        y=y_f_above37,
                        mode="lines",
                        marker_color="rgb(255, 0, 0)",
                        name="Female >37",
                    ),
                    row=1,
                    col=1,
                )

                text_annotation = "R-squared >37 f =<br>p-value >37 f ="
                res_annotation = (
                    f"{r_squared_female_a37:.5f}<br>{p_value_female_a37:.5f}"
                )

                fig.add_annotation(
                    x=1.25,
                    y=-0.05,
                    align="right",
                    text=text_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )
                fig.add_annotation(
                    x=1.35,
                    y=-0.05,
                    align="right",
                    text=res_annotation,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                )

                fig.add_trace(
                    go.Scatter(
                        x=column1_female[column1_female_bellow37],
                        y=y_f_bellow37,
                        mode="lines",
                        marker_color="rgb(255, 153, 153)",
                        name="Female <=37",
                    ),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            autosize=False,
            width=800,
            height=600,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            legend_title="Study",
        )

        if loglog:
            if savefig:
                self.save_figure(
                    fig,
                    column1,
                    column2,
                    typename="_scatterplot_gender_AgeSeparated_loglog.png",
                )
        else:
            if savefig:
                self.save_figure(
                    fig,
                    column1,
                    column2,
                    typename="_scatterplot_gender_AgeSeparated.png",
                )
        fig.show()
        return None

    def get_scatterplot_gender_ageSeparated_linReg_constraint(
        self,
        column1,
        column2,
        color_s,
        gender_needed,
        gender_investigated,
        reference_female,
        reference_male,
        savefig=False,
    ):
        """
        Get scatterplot of 2 specified columns, with linReg separated according to symbol_s and separated for <38 and >38
        Args:
            column1 (pd.Series): first column to plot
            column2 (pd.Series): second column to plot
            color_s (pd.Series): third column to categorize the scatterplot
            gender_needed (pd.Series): fourth column to distinguishe the markers in the plot and create separate linregs
            savefig: boolean if figure should be saved
        """

        columns = np.transpose(np.vstack((column1, column2)))
        index = ~np.isnan(columns).any(axis=1)

        fig = px.scatter(
            x=columns[index][:, 0],
            y=columns[index][:, 1],
            color=color_s[index],
            symbol=gender_needed[index],
            title=self.name,
        )

        if gender_investigated == "Female" or gender_investigated == "Male":
            if gender_investigated == "Female":
                marker_color1 = "rgb(255, 0, 0)"
                marker_color2 = "rgb(255, 153, 153)"
                ref = reference_female

            if gender_investigated == "Male":
                marker_color1 = "rgb(0, 0, 255)"
                marker_color2 = "rgb(153, 153, 255)"
                ref = reference_male

            x = np.array(
                columns[index][:, 0][
                    gender_needed[index] == gender_investigated[index]
                ][columns[index][:, 0] >= 37]
            )
            y = np.array(
                columns[index][:, 1][
                    gender_needed[index] == gender_investigated[index]
                ][columns[index][:, 1] >= 37]
            )
            data = np.stack((x, y), axis=1)
            data = data[~np.isnan(data).any(axis=1), :]

            intercept, slope = self.linear_regression_constraint(
                data[:, 0], data[:, 1], ref
            )
            fig.add_trace(
                go.Scatter(
                    x=column1[gender_needed == gender_investigated][column1 >= 37],
                    y=slope
                    * column1[gender_needed == gender_investigated][column1 >= 37]
                    + intercept,
                    mode="lines",
                    marker_color=marker_color1,
                    name=gender_investigated + " >=37",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=column1[gender_needed == gender_investigated][column1 <= 37],
                    y=len(column1[gender_needed == gender_investigated]) * [ref[0]],
                    mode="lines",
                    marker_color=marker_color2,
                    name=gender_investigated + " <37",
                ),
                row=1,
                col=1,
            )

            y_pred = intercept + slope * data[:, 0]
            residuals = data[:, 1] - y_pred
            ss_res = (residuals**2).sum()
            ss_tot = ((data[:, 1] - data[:, 1].mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot

            text_annotation = "R² " + gender_investigated + " = "
            res_annotation = f"{r2:.3f}"

            fig.add_annotation(
                x=0.93,
                y=1.0,
                align="right",
                text=text_annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
            )
            fig.add_annotation(
                x=1.0,
                y=1.0,
                align="right",
                text=res_annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
            )

        if gender_investigated == "Both":
            x_f = np.array(column1[gender_needed == "Female"][column1 >= 37])
            y_f = np.array(column2[gender_needed == "Female"][column1 >= 37])
            data_f = np.stack((x_f, y_f), axis=1)
            data_f = data_f[~np.isnan(data_f).any(axis=1), :]

            x_m = np.array(column1[gender_needed == "Male"][column1 >= 37])
            y_m = np.array(column2[gender_needed == "Male"][column1 >= 37])
            data_m = np.stack((x_m, y_m), axis=1)
            data_m = data_m[~np.isnan(data_m).any(axis=1), :]

            intercept_f, slope_f = self.linear_regression_constraint(
                data_f[:, 0], data_f[:, 1], reference_female
            )
            fig.add_trace(
                go.Scatter(
                    x=column1[gender_needed == "Female"][column1 >= 37],
                    y=slope_f * column1[gender_needed == "Female"][column1 >= 37]
                    + intercept_f,
                    mode="lines",
                    marker_color="rgb(255, 0, 0)",
                    name="Female >=37",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=column1[gender_needed == "Female"][column1 <= 37],
                    y=len(column1[gender_needed == "Female"]) * [reference_female[0]],
                    mode="lines",
                    marker_color="rgb(255, 153, 153)",
                    name="Female <37",
                ),
                row=1,
                col=1,
            )

            intercept_m, slope_m = self.linear_regression_constraint(
                data_m[:, 0], data_m[:, 1], reference_male
            )
            fig.add_trace(
                go.Scatter(
                    x=column1[gender_needed == "Male"][column1 >= 37],
                    y=slope_m * column1[gender_needed == "Male"][column1 >= 37]
                    + intercept_m,
                    mode="lines",
                    marker_color="rgb(0, 0, 255)",
                    name="Male >=37",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=column1[gender_needed == "Male"][column1 <= 37],
                    y=len(column1[gender_needed == "Male"]) * [reference_male[0]],
                    mode="lines",
                    marker_color="rgb(153, 153, 255)",
                    name="Male <37",
                ),
                row=1,
                col=1,
            )

            delta_y_m = (slope_m / reference_male[0]) * 100  # 1 years
            delta_y_f = (slope_f / reference_female[0]) * 100

            y_pred_f = intercept_f + slope_f * data_f[:, 0]
            residuals_f = data_f[:, 1] - y_pred_f
            ss_res_f = (residuals_f**2).sum()
            ss_tot_f = ((data_f[:, 1] - data_f[:, 1].mean()) ** 2).sum()
            r2_f = 1 - ss_res_f / ss_tot_f

            y_pred_m = intercept_m + slope_m * data_m[:, 0]
            residuals_m = data_m[:, 1] - y_pred_m
            ss_res_m = (residuals_m**2).sum()
            ss_tot_m = ((data_m[:, 1] - data_m[:, 1].mean()) ** 2).sum()
            r2_m = 1 - ss_res_m / ss_tot_m

            text_annotation = (
                "%Chg/year >=37 f   = <br>%Chg/year >=37 m = <br>%R² f = <br>%R² m = "
            )
            res_annotation = (
                f"{delta_y_f:.3f}<br>{delta_y_m:.3f}<br>{r2_f:.3f}<br>{r2_m:.3f}"
            )

            fig.add_annotation(
                x=0.93,
                y=1.0,
                align="right",
                text=text_annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
            )
            fig.add_annotation(
                x=1.0,
                y=1.0,
                align="right",
                text=res_annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
            )

        fig.update_layout(
            autosize=False,
            width=800,
            height=600,
            xaxis_title=column1.name,
            yaxis_title=column2.name,
            legend_title="Study",
        )
        if savefig:
            column_1_str = (
                column1.name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace(":", "")
                .replace("/", "_")
                .replace("²", "2")
            )
            column_2_str = (
                column2.name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace(":", "")
                .replace("/", "_")
                .replace("²", "2")
            )
            fig_name = f"{column_1_str}-{column_2_str}"
            fig.write_image(
                f"../02_OUTPUT/{fig_name}_scatterplot_gender_ageSeparated_constraint.png"
            )

        fig.show()
        return None

    def get_hatched_histogram(self, column1, pattern_shape):
        """get hatched histogram with a wanted column and pattern shape (secondary descriptor)

        Args:
            column1 (str): name of the column to plot (e.g. sex)
            pattern_shape (str): name of the secondary descriptor to plot (e.g. smoker yes/no)
        """
        fig = px.histogram(
            self.df,
            x=column1,
            color=pattern_shape,
            # pattern_shape=,
            title=self.name,
        )
        fig.update_layout(
            autosize=False, width=500, height=500, xaxis_title=column1.name
        )
        fig.show()
        return None

    def four_subplots(self, matrix1, matrix2, matrix3, matrix4):
        fig = make_subplots(rows=2, cols=2, start_cell="top-left")
        fig.add_trace(
            go.Scatter(x=self.df[matrix1[0]], y=self.df[matrix1[1]], mode="markers"),
            row=1,
            col=1,
        )
        fig.update_layout(
            xaxis_title=self.df[matrix1[0]].name, yaxis_title=self.df[matrix1[1]].name
        )

        fig.add_trace(
            go.Scatter(x=self.df[matrix2[0]], y=self.df[matrix2[1]], mode="markers"),
            row=1,
            col=2,
        )
        fig.update_layout(
            xaxis_title=self.df[matrix2[0]].name, yaxis_title=self.df[matrix2[1]].name
        )

        fig.add_trace(
            go.Scatter(x=self.df[matrix3[0]], y=self.df[matrix3[1]], mode="markers"),
            row=2,
            col=1,
        )
        fig.update_layout(
            xaxis_title=self.df[matrix3[0]].name, yaxis_title=self.df[matrix3[1]].name
        )

        fig.add_trace(
            go.Scatter(x=self.df[matrix4[0]], y=self.df[matrix4[1]], mode="markers"),
            row=2,
            col=2,
        )

        # Update xaxis properties
        fig.update_xaxes(title_text=self.df[matrix1[0]].name, row=1, col=1)
        fig.update_xaxes(title_text=self.df[matrix2[0]].name, row=1, col=2)
        fig.update_xaxes(title_text=self.df[matrix3[0]].name, row=2, col=1)
        fig.update_xaxes(title_text=self.df[matrix4[0]].name, row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text=self.df[matrix1[1]].name, row=1, col=1)
        fig.update_yaxes(title_text=self.df[matrix2[1]].name, row=1, col=2)
        fig.update_yaxes(title_text=self.df[matrix3[1]].name, row=2, col=1)
        fig.update_yaxes(title_text=self.df[matrix4[1]].name, row=2, col=2)

        fig.add_annotation(
            x=-0.1,
            y=-0.22,
            text=f"Generated by {self.originator_s} on {self.current_datetime}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            bordercolor="grey",
            borderwidth=0.1,
            xanchor="left",
            font_size=10,
        )

        # fig.add_annotation(
        #     x=1,
        #     y=-0.22,
        #     text=f"Raw filepath: {self.filepath}",
        #     align="right",
        #     showarrow=False,
        #     xref="paper",
        #     yref="paper",
        #     bordercolor="grey",
        #     borderwidth=0.1,
        #     xanchor="right",
        #     font_size=10,
        # )

        fig.update_layout(
            height=self.height,
            width=self.height,  # make it square
            title_text="Study " + self.name,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
        )
        if self.showfig:
            fig.show()
        if self.savefig:
            self.save_figure(fig, None, None, typename="Hatched_histogram.png")

    def three_subplots(
        self, column1_1, column1_2, column2_1, column2_2, column3_1, column3_2
    ):
        fig = make_subplots(
            rows=2,
            cols=3,
            start_cell="top-left",
            specs=[[{"rowspan": 2, "colspan": 2}, None, {}], [None, None, {}]],
            subplot_titles=("Radar plot", "Bone sample", "Trendline plot"),
        )

        fig.add_trace(
            go.Scatter(x=column1_1, y=column1_2, mode="markers"),
            row=1,
            col=1,
        )
        # fig.update_layout(xaxis_title=self.df[matrix1[0]].name, yaxis_title=self.df[matrix1[1]].name)

        fig.add_trace(
            go.Scatter(x=column2_1, y=column2_2, mode="markers"),
            row=1,
            col=3,
        )
        # fig.update_layout(xaxis_title=self.df[matrix2[0]].name, yaxis_title=self.df[matrix2[1]].name)

        fig.add_trace(
            go.Scatter(x=column3_1, y=column3_2, mode="markers"),
            row=2,
            col=3,
        )
        # fig.update_layout(xaxis_title=self.df[matrix3[0]].name, yaxis_title=self.df[matrix3[1]].name)

        # Update xaxis properties
        fig.update_xaxes(title_text=column1_1.name, row=1, col=1)
        # fig.update_xaxes(title_text=column2_1.name, row=1, col=2)
        fig.update_xaxes(title_text=column3_1.name, row=2, col=3)

        # Update yaxis properties
        fig.update_yaxes(title_text=column1_2.name, row=1, col=1)
        # fig.update_yaxes(title_text=column2_2.name, row=1, col=2)
        fig.update_yaxes(title_text=column3_2.name, row=2, col=3)

        """
        fig.add_annotation(
            x=-0.1,
            y=-0.22,
            text=f"Generated by {self.originator_s} on {self.current_datetime}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            bordercolor="grey",
            borderwidth=0.1,
            xanchor="left",
            font_size=10,
        )

        fig.add_annotation(
            x=1,
            y=-0.22,
            text=f"Raw filepath: {self.filepath}",
            align="right",
            showarrow=False,
            xref="paper",
            yref="paper",
            bordercolor="grey",
            borderwidth=0.1,
            xanchor="right",
            font_size=10,
        )"""

        fig.update_layout(
            height=self.height,
            width=self.width,
            # title_text="Study " + self.name,
            # legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            showlegend=False,
        )
        if self.showfig:
            fig.show()

    def three_subplots_images(
        self,
        image_title_01,
        image_title_02,
        image_title_03,
    ):
        # ../02_OUTPUT/{fig_name}_scatterplot_patient_asessment.png

        fig = make_subplots(
            rows=2,
            cols=3,
            start_cell="top-left",
            specs=[[{"rowspan": 2, "colspan": 2}, None, {}], [None, None, {}]],
            # subplot_titles=("Radar plot", "Bone sample", "Trendline plot"),
            column_widths=[1, 1, 1],
        )
        # horizontal_spacing=0.1)

        # Load PNG files using PIL
        img1 = np.array(
            Image.open("../02_OUTPUT/" + image_title_01 + ".png").convert("RGB")
        )  # .resize((200, 150))
        img2 = np.array(
            Image.open("../02_OUTPUT/" + image_title_02 + ".png").convert("RGB")
        )  # .resize((100, 100))
        img3 = np.array(
            Image.open("../02_OUTPUT/" + image_title_03 + ".png").convert("RGB")
        )  # .resize((200, 200))

        # Resize images to target sizes using the Pillow library
        img1 = np.array(Image.fromarray(img1))
        img2 = np.array(Image.fromarray(img2))
        img3 = np.array(Image.fromarray(img3))

        # Add the PNG files to the subplots
        fig.add_trace(go.Image(z=img1), row=1, col=1)
        fig.add_trace(go.Image(z=img2), row=1, col=3)
        fig.add_trace(go.Image(z=img3), row=2, col=3)

        fig.update_layout(
            height=self.height,
            width=self.width,
            # title_text="Study " + self.name,
            # legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            showlegend=False,
        )
        # Remove the x- and y-axes of each image
        for i in range(1, 3):
            for j in range(1, 4):
                fig.update_xaxes(
                    showgrid=False, zeroline=False, showticklabels=False, row=i, col=j
                )
                fig.update_yaxes(
                    showgrid=False, zeroline=False, showticklabels=False, row=i, col=j
                )
        fig.show()
        if self.savefig:
            self.save_figure(fig, None, None, typename="Three_subplots_image.png")

    def two_subplots(self, matrix1, matrix2):
        fig = make_subplots(rows=1, cols=2, start_cell="top-left")
        fig.add_trace(
            go.Scatter(x=self.df[matrix1[0]], y=self.df[matrix1[1]], mode="markers"),
            row=1,
            col=1,
        )
        fig.update_layout(
            xaxis_title=self.df[matrix1[0]].name, yaxis_title=self.df[matrix1[1]].name
        )

        fig.add_trace(
            go.Scatter(x=self.df[matrix2[0]], y=self.df[matrix2[1]], mode="markers"),
            row=1,
            col=2,
        )
        fig.update_layout(
            xaxis_title=self.df[matrix2[0]].name, yaxis_title=self.df[matrix2[1]].name
        )

        # Update xaxis properties
        fig.update_xaxes(title_text=self.df[matrix1[0]].name, row=1, col=1)
        fig.update_xaxes(title_text=self.df[matrix2[0]].name, row=1, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text=self.df[matrix1[1]].name, row=1, col=1)
        fig.update_yaxes(title_text=self.df[matrix2[1]].name, row=1, col=2)

        fig.add_annotation(
            x=-0.1,
            y=-0.35,
            text=f"Generated by {self.originator_s} on {self.current_datetime}",
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            bordercolor="grey",
            borderwidth=0.1,
            xanchor="left",
            font_size=10,
        )

        fig.add_annotation(
            x=1,
            y=-0.35,
            text=f"Raw filepath: {self.filepath}",
            align="right",
            showarrow=False,
            xref="paper",
            yref="paper",
            bordercolor="grey",
            borderwidth=0.1,
            xanchor="right",
            font_size=10,
        )

        fig.update_layout(
            height=self.height,
            width=self.width,
            title_text="Study " + self.name,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5
            ),
        )
        if self.showfig:
            fig.show()
        return fig
