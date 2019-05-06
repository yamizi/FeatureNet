import xml.etree.ElementTree as ET

nb_cells = 10
nb_blocks = 1

tree = ET.parse("C:\\Users\salah.ghamizi\Desktop\main_1block_light.xml")
root = tree.getroot()

child = root.getchildren()
feature_tree = child[0]
constraints = child[1]

feature_tree_raw = feature_tree.text

cellIndex = feature_tree_raw.find(":o Block[k]_Element[i]")
cell = feature_tree_raw[cellIndex:]
all_cells = "\t\t\t\t\t".join([cell.replace("[i]",str(i)) for i in range(nb_cells)])

feature_tree_raw = feature_tree_raw[:cellIndex] + all_cells


blockIndex = feature_tree_raw.find(":o Block[k](Block[k])")
block = feature_tree_raw[blockIndex:]
all_blocks = "\t\t\t\t".join([block.replace("[k]",str(k)) for k in range(nb_blocks)])
feature_tree_raw = feature_tree_raw[:blockIndex] + all_blocks

feature_tree.text  = feature_tree_raw

constraints_raw = constraints.text.split("\n")

constraints_full = ""
constraint_id = 1
for c in range(len(constraints_raw)):
    
    constraint = constraints_raw[c].split(":")
    if len(constraint) < 2:
        continue
    if constraint[1].find("[k]")==-1:
        constraints_full ="{}\n{}".format(constraints_full, c)
    elif constraint[0].find("CLC")!=-1:
        pass
    else:
        for k in range(nb_blocks):
            if constraint[1].find("[k+1]")!=-1 and k+1==nb_blocks:
                continue

            if constraint[1].find("[i]")==-1:
                replaced = constraint[1].replace("[k]", str(k)).replace("[k+1]", str(k+1))
                #constraints_full ="{}\n{}{}0:{}".format(constraints_full, constraint[0], str(k), replaced)
                constraints_full ="{}\nC{}:{}".format(constraints_full, constraint_id, replaced)
                constraint_id = constraint_id+1
            else:
                for i in range(nb_cells):
                    if constraint[1].find("[i+1]")!=-1 and i+1==nb_cells:
                        continue
                    replaced = constraint[1].replace("[k]", str(k)).replace("[k+1]", str(k+1)).replace("[i]", str(i)).replace("[i+1]", str(i+1))
                    #constraints_full ="{}\n{}{}{}:{}".format(constraints_full, constraint[0], str(k),i, replaced)
                    constraints_full ="{}\nC{}:{}".format(constraints_full, constraint_id, replaced)
                    constraint_id = constraint_id+1
constraints.text = constraints_full+"\n"

tree.write("C:\\Users\salah.ghamizi\Desktop\main_1block_10_cells.xml", encoding="UTF-8", xml_declaration=True)