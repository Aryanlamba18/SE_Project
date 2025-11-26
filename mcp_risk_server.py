from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np

mcp = FastMCP("RiskEngine")

@mcp.tool()
def calculate_volatility(prices: list[float]) -> float:
    return float(pd.Series(prices).pct_change().std() * np.sqrt(252))

if __name__ == "__main__":
    mcp.run()