def parse_input_file(path):
    number_of_locations = None
    belts = []
    orders = {}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Number of locations: 15
            if line.startswith("Number of locations"):
                number_of_locations = int(line.split(":")[1])

            # Belts P1, P2, P3: 2, 5, 9
            elif line.startswith("Belts"):
                parts = line.split(":")[1]
                belts = [int(x.strip()) for x in parts.split(",")]

            # Order O1: 1, 1, 3, 3, 2
            elif line.startswith("Order"):
                key_part, nums_part = line.split(":")
                order_name = key_part.split()[1]   # "O1", "O2", ...
                nums = [int(x.strip()) for x in nums_part.split(",")]
                orders[order_name] = nums

    return {
        "number_of_locations": number_of_locations,
        "belts": belts,
        "orders": orders
    }


# Example usage:
data = parse_input_file("data.txt")
print(data)
