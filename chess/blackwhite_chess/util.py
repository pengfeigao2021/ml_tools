
def recursion(point, points, classified, index):
    if index not in classified:
        classified[index] = []

    cur_x, cur_y = point
    for x in range(cur_x-1,cur_x+2):
        for y in range(cur_y-1,cur_y+2):
            # print((x,y)
            is_exist = False
            if (x,y) == point: continue
            for values in classified.values():
                if (x,y) in values: 
                    is_exist = True
                    break

            if not is_exist and (x,y) in points:
                classified[index].append((x,y))
                recursion((x,y), points, classified, index)

def connected_domain(points):
    classified = {}
    index = 0
    for point in points:
        is_exist = False
        for values in classified.values():
            if point in values: 
                is_exist = True
                break
        if is_exist: continue

        # print("index: %d" % index
        # print(
        classified[index] = [point]
        recursion(point, points, classified, index)
        index += 1
    return classified



def is_line(p1, p2, p3):
    y1_diff = p1[1]-p2[1]
    x1_diff = p1[0]-p2[0]
    y2_diff = p3[1]-p1[1]
    x2_diff = p3[0]-p1[0]
    if x1_diff == 0:
        if x2_diff == 0:
            return True
        else:
            return False
    else:
        if x2_diff == 0:
            return False
        else:
            return (float(y1_diff)/x1_diff) == (float(y2_diff)/x2_diff)

def five_in_a_row(points):
    for p1 in points:
        for p2 in points:
            in_line_list = [p1,p2]
            if p1 == p2: continue
            for p3 in points:
                if p3 == p1 or p3 == p2: continue
                if is_line(p1, p2, p3):
                    in_line_list.append(p3)
            if len(in_line_list) >= 5:
                domains = connected_domain(in_line_list)
                if len(domains) == 1:
                    return in_line_list
    return None

def test_five_in_a_row():
    points = []
    points.append((1,0))
    points.append((1,1))
    points.append((1,2))
    points.append((1,3))
    points.append((1,4))

    points.append((0,2))
    points.append((2,1))

    print(five_in_a_row(points))

    print("\n11111")
    print("00000")
    print("00000")
    print("00000")
    print("00000")
    points = []
    points.append((1,0))
    points.append((2,0))
    points.append((3,0))
    points.append((4,0))
    points.append((5,0))

    points.append((5,1))
    points.append((3,1))

    print(five_in_a_row(points))

    print("\n10000")
    print("01000")
    print("00100")
    print("00010")
    print("00001")
    points = []
    points.append((1,1))
    points.append((2,2))
    points.append((3,3))
    points.append((4,4))
    points.append((5,5))

    points.append((5,4))
    points.append((3,2))

    print(five_in_a_row(points))

if __name__ == "__main__":
    points = []
    points.append((1,0))
    points.append((1,1))
    points.append((1,2))
    points.append((0,2))
    points.append((1,3))
    points.append((2,1))
    points.append((1,4))


    # Test finding connected domains through recursion
    """
    classified = connected_domain(points)
    for i in classified:
        print(i, classified[i]
    """

    # Test find five in a row
    test_five_in_a_row()

