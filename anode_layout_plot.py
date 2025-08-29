import matplotlib.pyplot as plt
import matplotlib.patches as patches

def layout_of_anode():
    # Parameters for rectangles
    rect_width = 45
    rect_height = 6.6
    gap = 3
    num_rectangles = 2  # Number of rectangles per row
    rows = 8  # Number of rows
    row_names = ["AB", "CD", "EF", "GH", "IJ", "KL", "MN", "OP"]
    row_names = row_names[::-1]  # Reverse to match top-to-bottom order

    # Color mapping from the uploaded image
    color_mapping = {
        "OP1": "#B0B0B0",
        "OP0": "#B0B0B0",
        "CD1": "#B0B0B0",
        "CD0": "#B0B0B0",
        "GH1": "#B0B0B0",
        "GH0": "#B0B0B0",
        "KL1": "#B0B0B0",
        "KL0": "#B0B0B0",
    }

    def get_color(row_name):
        return color_mapping.get(row_name, "#c9ac77")  # Default to grey if not in mapping

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the rectangles
    for row in range(rows):
        row_name = row_names[row]
        for i in range(num_rectangles):
            x =  (num_rectangles - 1 - i) * (rect_width + gap)  # Position based on gap and width
            y = row * (rect_height + gap)  # Position based on row and gap
            row_name_array = f"{row_name}{i}"
            print(f"Drawing {row_name_array} at ({x}, {y})")
            # Add rectangle to the plot
            rect = patches.Rectangle((x, y), rect_width, rect_height, linewidth=1,
                                    edgecolor='black', facecolor=get_color(row_name_array))
            ax.add_patch(rect)

            # Add label in the middle of the rectangle
            label_x = x + rect_width / 2
            label_y = y + rect_height / 2
            ax.text(label_x, label_y, f"{row_name_array}", fontsize=25, ha='center', va='center', color='black')

    # Set limits and aspect ratio
    ax.set_xlim(-5, num_rectangles * (rect_width + gap))
    ax.set_ylim(-5, rows * (rect_height + gap))
    ax.set_aspect('equal', adjustable='datalim')

    # Remove axes for a cleaner look
    ax.axis('off')

    # Show the plot
    return fig, ax

def layout_of_anode_fine():
    # Parameters for rectangles
    rect_width = 45
    rect_height = 6.6
    gap = 3
    num_rectangles = 2  # Number of rectangles per row
    rows = 48  # Number of rows
    row_names = [f"Row{row}" for row in range(rows)]  # Generate row names dynamically

    # Color mapping (optional, can be customized)
    color_mapping = {
        "Row31": "blue",
        
        "Row41": "orange",
        
        "Row51": "cyan",
        
        "Row401": "green",
        
        "Row421": "red",
        
        # Add more mappings if needed
    }


    def get_color(row_name):
        return color_mapping.get(row_name, "#B0B0B0")  # Default to grey if not in mapping

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 55))  # Adjust height to fit 55 rows

    # Draw the rectangles
    for row in range(rows):
        row_name = row_names[row]
        for i in range(num_rectangles):
            x = (num_rectangles - 1 - i) * (rect_width + gap)  # Position based on gap and width
            y = row * (rect_height + gap)  # Position based on row and gap
            row_name_array = f"{row_name}{i}"
            print(f"Drawing {row_name_array} at ({x}, {y})")
            # Add rectangle to the plot
            rect = patches.Rectangle((x, y), rect_width, rect_height, linewidth=1,
                                      edgecolor='black', facecolor=get_color(row_name_array))
            ax.add_patch(rect)

            # Add label in the middle of the rectangle
            label_x = x + rect_width / 2
            label_y = y + rect_height / 2
            if row_name_array in color_mapping:
                ax.text(label_x, label_y, f"{row+48+1}", fontsize=10, ha='center', va='center', color='black')
            #ax.text(label_x, label_y, f"{row_name_array}", fontsize=8, ha='center', va='center', color='black')

    # Set limits and aspect ratio
    ax.set_xlim(-5, num_rectangles * (rect_width + gap))
    ax.set_ylim(-5, rows * (rect_height + gap))
    ax.set_aspect('equal', adjustable='datalim')

    # Remove axes for a cleaner look
    ax.axis('off')

    # Show the plot
    return fig, ax

# Example usage
fig, ax = layout_of_anode_fine()
plt.show()
