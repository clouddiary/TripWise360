import asyncio
from mcp_client import get_all_tools, aviation_mcp_call


# async def test_schedule():
#     """Test flight_arrival_departure_schedule with correct parameter names."""
#     print("\n--- Testing flight_arrival_departure_schedule (JFK departures) ---")
#     try:
#         result = await aviation_mcp_call(
#             "flight_arrival_departure_schedule",
#             {"airport_iata_code": "JFK", "schedule_type": "departure"}
#         )
#         print(result)
#     except Exception as e:
#         print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(get_all_tools())