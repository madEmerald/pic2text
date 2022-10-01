from PIL import Image, ImageDraw


class Component:
    def __init__(self, x, y):
        self._pixels = [(x, y)]
        self._min_x = x
        self._max_x = x
        self._min_y = y
        self._max_y = y

    def add(self, x, y):
        self._pixels.append((x, y))
        self._min_x = min(self._min_x, x)
        self._max_x = max(self._max_x, x)
        self._min_y = min(self._min_y, y)
        self._max_y = max(self._max_y, y)

    # корзина для распознаваемых образов
    def basket(self):
        return (self._min_x, self._min_y), (self._max_x, self._max_y)


def find_components(im):
    width, height = im.size
    components = {}
    pixel_component = [[0 for __ in range(height)] for _ in range(width)]
    equivalences = {}
    components_number = 0

    # первый проход и поиск компонентов
    for i in range(width):
        for j in range(height):
            # проверка на чёрный пиксел
            if im.getpixel((i, j)) == (0, 0, 0, 255):
                # поиск пикселов, различающихся по значению
                upper_component = pixel_component[i - 1][j] if i > 0 else 0
                left_component = pixel_component[i][j - 1] if j > 0 else 0

                min_component, max_component = sorted((upper_component, left_component))

                if min_component > 0:
                    new_component = min_component

                    if min_component != max_component:
                        if max_component in equivalences:
                            equivalences[max_component].add(new_component)
                        else:
                            equivalences[max_component] = {new_component}

                elif max_component > 0:
                    new_component = max_component
                else:
                    components_number += 1
                    new_component = components_number

                pixel_component[i][j] = new_component

    # присваивание всем эквивалентным компонентам одинакового значения
    for i in range(width):
        for j in range(height):
            r = pixel_component[i][j]
            if r > 0:
                while r in equivalences:
                    r = min(equivalences[r])

                if r not in components:
                    components[r] = Component(i, j)
                else:
                    components[r].add(i, j)

    return list(components.values())


def main():
    im = Image.open(r"example.png")
    components = find_components(im)
    draw = ImageDraw.Draw(im)

    for r in components:
        draw.rectangle(r.basket(), outline=(255, 0, 0))
    del draw

    im.show()  # проверка
    output = open("output.png", "wb")
    im.save(output)
    output.close()


if __name__ == "__main__":
    main()
