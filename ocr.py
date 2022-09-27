import sys
from PIL import Image, ImageDraw


class Component():
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
    #корзина для распознаваемых образов
    def basket(self):
        return [(self._min_x, self._min_y), (self._max_x, self._max_y)]


def find_components(im):
    width, height  = im.size
    components = {}
    pixel_component = [[0 for y in range(height)] for x in range(width)]
    equivalences = {}
    n_components = 0

    # первый проход и поиск компонентов
    for x in range(width):
        for y in range(height):
            # проверка на чёрный пиксел
            if im.getpixel((x, y)) == (0, 0, 0, 255):
                # поиск пикселов, различающихся по значению
                component_n = pixel_component[x - 1][y] if x > 0 else 0
                component_w = pixel_component[x][y - 1] if y > 0 else 0

                max_component = max(component_n, component_w)

                if max_component > 0:
                    new_component = min(filter(lambda i: i > 0, (component_n, component_w)))
                    if max_component > new_component:
                        if max_component in equivalences:
                            equivalences[max_component].add(new_component)
                        else:
                            equivalences[max_component] = set((new_component,))
                else:
                    n_components += 1
                    new_component = n_components

                pixel_component[x][y] = new_component

    #присваивание всем эквивалентным компонентам одинакового значения
    for x in range(width):
        for y in range(height):
                r = pixel_component[x][y]
                if r > 0:
                    while r in equivalences:
                        r = min(equivalences[r])

                    if not r in components:
                        components[r] = Component(x, y)
                    else:
                        components[r].add(x, y)

    return list(components.itervalues())


def main():
    im = Image.open(r"c:\users\perso\OCR\img\test.png")
    components = find_components(im)
    draw = ImageDraw.Draw(im)

    for r in components:
        draw.rectangle(r.basket(), outline=(255, 0, 0))
    del draw

    #im.show() #проверка
    output = file("output.png", "wb")
    im.save(output)
    output.close()

if __name__ == "__main__":
    main()